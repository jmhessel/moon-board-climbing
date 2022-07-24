'''
Can we predict good vs. bad from the existing annotations?
accelerate launch train_reddit.py binary_instance.json --t5_model t5-large --use_accelerate 1
'''
import argparse

import collections
import tqdm
import json
import numpy as np
import os
import sklearn
import transformers
import accelerate
import tempfile
import torch
import random
import subprocess
import scipy
import sklearn.metrics
import pprint

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('binary_instances')

    parser.add_argument('--t5_model',
                        default='t5-small',
                        help='which t5 model to use?')

    parser.add_argument('--batch_size',
                        default=32,
                        type=int)

    parser.add_argument('--workers_dataloader',
                        type=int,
                        default=8)

    parser.add_argument('--n_epochs',
                        type=int,
                        default=7)

    parser.add_argument('--lr',
                        type=float,
                        default=.0001)

    parser.add_argument('--use_accelerate',
                        type=int,
                        default=0,
                        help='if this flag is set, we will use huggingface accelerate intsead of dataparallel')

    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help='how many steps for gradient accumulation')

    parser.add_argument('--patience',
                        type=int,
                        default=5)

    parser.add_argument('--skip_save',
                        type=int,
                        default=0)

    parser.add_argument('--use_adafactor',
                        type=int,
                        default=0)

    parser.add_argument('--verbose',
                        type=int,
                        default=0,
                        help='if verbose, during validation, we will print out the instances with the top predictions for different classes.')

    args = parser.parse_args()
    args.output_path = 'generator_valacc={:.5f}' + '~model=' + '{}'.format(args.t5_model.replace('/', '*')) + '~lr={}'.format(args.lr) + '.pt'
    if not args.skip_save:
        print('saving to {}'.format(args.output_path))

    return args


class T5Dataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer, args):
        self.data = data
        self.tokenizer = tokenizer
        self.args = args

    def __getitem__(self, idx):
        res = self.tokenizer(self.data[idx]['input'], truncation=True, max_length=512)
        res['labels'] = self.tokenizer(self.data[idx]['target']).input_ids
        return res

    def __len__(self):
        return len(self.data)


def batch_to_device(batch, mode, args):
    input_ids, attention_mask, labels = batch['input_ids'], batch['attention_mask'], batch['labels']
    if not args.use_accelerate or mode == 'val':
        input_ids, attention_mask, labels = map(
            lambda x: x.to(args.device),
            [input_ids, attention_mask, labels])

    return dict(zip(['input_ids', 'attention_mask', 'labels'],
                    [input_ids, attention_mask, labels]))


def main():
    args = parse_args()
    assert args.use_accelerate, 'Only accelerate supported.'

    np.random.seed(1)
    random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)

    with open(args.binary_instances) as f:
        data = json.load(f)
        train_d, val_d, test_d = data['train'], data['val'], data['test']

    print('train/val/test datapoints: {}/{}/{}'.format(*map(len, [train_d, val_d, test_d])))

    # baseline accuracies...

    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    accelerator = accelerate.Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)
    mainproc = accelerator.is_local_main_process

    n_labels = len(set([d['target'] for d in train_d]))
    print('{} labels'.format(n_labels))

    tokenizer = transformers.T5TokenizerFast.from_pretrained(args.t5_model)
    # get one of the label idxs; used for precision computation if n_labels==2
    label_idx = [x[0] for x in tokenizer(sorted(list(set([d['target'] for d in train_d])))).input_ids]
    print('label idxs {}'.format(label_idx))
    # grab one label idx for computing precision if n_labels = 2
    label_idx = label_idx[0]

    train_loader, val_loader, test_loader = map(
        lambda x: T5Dataset(x, tokenizer, args),
        [train_d, val_d, test_d])

    model = transformers.T5ForConditionalGeneration.from_pretrained(args.t5_model)

    data_collator = transformers.DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=-100,
        return_tensors='pt'
    )

    train_loader = torch.utils.data.DataLoader(
        train_loader, shuffle=True, collate_fn=data_collator, batch_size=args.batch_size
    )

    val_loader = torch.utils.data.DataLoader(
        val_loader, shuffle=True, collate_fn=data_collator, batch_size=args.batch_size
    )

    test_loader = torch.utils.data.DataLoader(
        test_loader, shuffle=False, collate_fn=data_collator, batch_size=args.batch_size
    )

    if not args.use_accelerate and torch.cuda.device_count() > 1:
        print('Lets use', torch.cuda.device_count(), 'GPUs!')
        model = torch.nn.DataParallel(model)

    if not args.use_accelerate:
        model.to(args.device)
    else:
        args.device = accelerator.device

    trainable_params = model.parameters()
    if not args.use_adafactor:
        optim = torch.optim.AdamW(trainable_params, lr=args.lr)
    else:
        optim = transformers.optimization.Adafactor(trainable_params, scale_parameter=False, relative_step=False, warmup_init=False, lr=args.lr)

    best_val_acc, not_improved_epoch = 0, 0

    if mainproc:
        tmpfile = tempfile.NamedTemporaryFile()
        print('using tempfile {}'.format(tmpfile.name))

    if args.use_accelerate:
        model, optim, train_loader = accelerator.prepare(model, optim, train_loader)

    for epoch in range(args.n_epochs):
        if mainproc:
            print('Epoch {}'.format(epoch))
        for mode in ['train', 'val']:
            if mode == 'train':
                model.train()
                bar = tqdm.tqdm(enumerate(train_loader), total=len(train_loader), disable=not mainproc)
            else:
                model.eval()
                bar = tqdm.tqdm(enumerate(val_loader), total=len(val_loader), disable=not mainproc)

            n, running_sum_loss, n_exs, running_sum_accs = 0, 0, 0, 0
            all_preds, all_labels = [], []

            pred2logits_and_labels = collections.defaultdict(list)

            for i, batch in bar:
                with torch.set_grad_enabled(mode=='train'):
                    with accelerator.accumulate(model):
                        batch = batch_to_device(batch, mode, args)
                        labels = batch['labels']
                        output = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['labels'])
                        loss = output['loss'].mean()

                        if mode == 'train':
                            if args.use_accelerate:
                                accelerator.backward(loss)
                            else:
                                loss.backward()
                            optim.step()
                            optim.zero_grad()

                        n += 1
                        n_exs += output['logits'].shape[0]
                        with torch.no_grad():
                            predictions = output['logits'][:, 0, :].argmax(-1).flatten().cpu().numpy()
                            accs = (labels[:, 0].cpu().flatten().numpy() == predictions).sum()
                        running_sum_loss += loss.cpu().detach().numpy()
                        running_sum_accs += accs

                        bar.set_description('loss = {:.6f} acc = {:.6f}'.format(running_sum_loss / n, running_sum_accs / n_exs))

            if mode == 'val' and mainproc:
                val_acc = running_sum_accs / n_exs
                if val_acc > best_val_acc:
                    print('{} is a better than than {}, saving weights!'.format(
                        val_acc,
                        best_val_acc))
                    best_val_acc = val_acc
                    if not args.skip_save:
                        if args.use_accelerate:
                            torch.save(
                                {'model_state_dict': accelerator.unwrap_model(model).state_dict(),
                                 'args': vars(args)},
                                tmpfile.name)
                        else:
                            try:
                                torch.save(
                                    {'model_state_dict': model.module.state_dict(),
                                     'optimizer_state_dict': optim.state_dict(),
                                     'args': vars(args)},
                                    tmpfile.name)
                            except:
                                torch.save(
                                    {'model_state_dict': model.state_dict(),
                                     'optimizer_state_dict': optim.state_dict(),
                                     'args': vars(args)},
                                    tmpfile.name)
                        not_improved_epoch = 0
                else:
                    not_improved_epoch += 1
        if not_improved_epoch == args.patience and not args.use_accelerate:
            print('Havent improved in {} epochs, breaking.'.format(not_improved_epoch))
            break

    accelerator.wait_for_everyone()
    if mainproc and not args.skip_save:
        args.output_path = args.output_path.format(best_val_acc)
        subprocess.call('cp {} {}'.format(tmpfile.name, args.output_path), shell=True)


if __name__ == '__main__':
    main()
