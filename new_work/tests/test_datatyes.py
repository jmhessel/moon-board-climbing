import unittest

import os, sys
import_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+'/types'
sys.path.append(import_path)

from hold import Hold
import hold

class TestHoldType(unittest.TestCase):

	def test_load_data(self):
		example_hold = Hold('website_format','A12')
		self.assertEqual(example_hold.row,12)
		self.assertEqual(example_hold.col,1)

	def test_climb_utilities(self):
		self.assertEqual(hold._character_to_int('A'),1)
		self.assertEqual(hold._character_to_int('a'),1)
		self.assertEqual(hold._character_to_int('b'),2)
		self.assertEqual(hold._int_to_char(1),'A')
		self.assertEqual(hold._int_to_char(2),'B')

	def test_website_format(self):
		for holdname in ['A12','A1','D8']:
			example_hold = Hold('website_format',holdname)
			self.assertEqual(example_hold.as_website_format(),holdname)

	def test_nn_format(self):
		hold_input = ['A1','A2','B1']
		hold_output = ['Aa','Ab','Ba']

		for holdname, formatted in zip(hold_input,hold_output):
			example_hold = Hold('website_format',holdname)
			self.assertEqual(example_hold.as_nn_format(),formatted)

	def test_invalid_input(self):
		self.assertRaises(ValueError,Hold,'website_format','A0')
		self.assertRaises(ValueError,Hold,'website_format','A99')
		self.assertRaises(ValueError,Hold,'website_format','A')
		self.assertRaises(IndexError,Hold,'website_format','')
		self.assertRaises(ValueError,Hold,'website_format',1)

from climb import Climb

class TestClimbType(unittest.TestCase):

	def test_load_data(self):
		example_climb_info = {'Grade': '8A', 'UserRating': 0, 'Moves': ['G2', 'J7', 'J8', 'D8', 'D10', 'A5', 'A13', 'F6', 'D16', 'C18']}
		example_climb = Climb('json',example_climb_info)
		self.assertEqual(example_climb.grade.as_font_grade(),'8A')
		self.assertEqual(example_climb.rating,0)

		# Check that the first hold was input properly
		self.assertEqual(example_climb.holds[0].row,2)
		self.assertEqual(example_climb.holds[0].col,7)
		self.assertEqual(len(example_climb.holds),10)

	def test_nn_string(self):
		example_climb_info = {'Grade': '8A', 'UserRating': 0, 'Moves': ['G2', 'J7', 'J8', 'D8', 'D10', 'A5', 'A13', 'F6', 'D16', 'C18']}
		example_climb = Climb('json',example_climb_info)
		self.assertEqual(
			example_climb.moves_nn_string(),
			'GbJgJhDhDjAeAmFfDpCr')

from climbset import Climbset

def new_climbset():
	example_climb_info = {'Grade': '8A', 'UserRating': 0, 'Moves': ['G2', 'J7', 'J8', 'D8', 'D10', 'A5', 'A13', 'F6', 'D16', 'C18']}
	example_climb = Climb('json',example_climb_info)
	example_climb_info2 = {'Grade': '7A', 'UserRating': 0, 'Moves': ['G3', 'J7', 'J8', 'D8', 'D10', 'A5', 'A13', 'F6', 'D16', 'C18']}
	example_climb2 = Climb('json',example_climb_info2)
	example_climb_info3 = {'Grade': '6C', 'UserRating': 0, 'Moves': ['G4', 'J7', 'J8', 'D8', 'D10', 'A5', 'A13', 'F6', 'D16', 'C18']}
	example_climb3 = Climb('json',example_climb_info3)
	example_climb_list = [example_climb,example_climb2,example_climb3]
	return Climbset(example_climb_list)

class TestClimbsetType(unittest.TestCase):

	def test_load_data(self):
		example_climbset = new_climbset()

		# Check all 3 climbs were imported
		self.assertEqual(len(example_climbset.climbs),3)

		# Check that the first move of the first climb was imported correctly
		first_move = example_climbset.climbs[0].holds[0].as_website_format()
		self.assertEqual(first_move,'G2')

		# Check all the grades were input right
		self.assertEqual(example_climbset.climbs[0].grade.as_font_grade(),'8A')
		self.assertEqual(example_climbset.climbs[1].grade.as_font_grade(),'7A')
		self.assertEqual(example_climbset.climbs[2].grade.as_font_grade(),'6C')

	# Test string generatino stuff
	def test_pre_format(self):
		example_climbset = new_climbset()
		self.assertEqual(
			example_climbset.pre_grade_string(),
			'11GbJgJhDhDjAeAmFfDpCr5GcJgJhDhDjAeAmFfDpCr3GdJgJhDhDjAeAmFfDpCr')

	def test_post_format(self):
		example_climbset = new_climbset()
		self.assertEqual(
			example_climbset.post_grade_string(),
			'GbJgJhDhDjAeAmFfDpCr11GcJgJhDhDjAeAmFfDpCr5GdJgJhDhDjAeAmFfDpCr3')

	def test_no_format(self):
		example_climbset = new_climbset()
		self.assertEqual(
			example_climbset.no_grade_string(),
			'GbJgJhDhDjAeAmFfDpCrGcJgJhDhDjAeAmFfDpCrGdJgJhDhDjAeAmFfDpCr')


if __name__ == '__main__':
	unittest.main(verbosity = 2)
