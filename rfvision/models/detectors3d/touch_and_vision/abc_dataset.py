#Copyright (c) Facebook, Inc. and its affiliates.
#All rights reserved.
#This source code is licensed under the license found in the
#LICENSE file in the root directory of this source tree.
from scipy.spatial.transform import Rotation as R
import os
from glob import glob
from tqdm import tqdm
import scipy.io as sio
import random
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
from rfvision.datasets.builder import DATASETS
from rfvision.datasets.pipelines import Compose
from .utils import point_loss
from .utils import chamfer_distance_mesh2pc



preprocess = transforms.Compose([
	transforms.Resize((256, 256)),
	transforms.ToTensor()
])

# class used for obtaining an instance of the dataset for training vision chart prediction
# to be passed to a pytorch dataloader
# input:
#	- classes: list of object classes used
# 	- set_type: the set type used
# 	- sample_num: the size of the point cloud to be returned in a given batch


@DATASETS.register_module()
class ABCVisionDataset(object):
	def __init__(self,
				 data_root,
				 anno_root,
				 pipeline=None,
				 classes = ['0001', '0002'],
				 num_grasps=1,
				 set_type='train',
				 num_samples=3000,
				 test_mode=False):

		# initialization of data locations
		self.CLASSES = classes
		self.test_mode = test_mode
		self.num_grasps=num_grasps
		self.surf_location = f'{data_root}/surfaces/'
		self.img_location = f'{data_root}/images/'
		self.touch_location = f'{data_root}/scene_info/'
		self.sheet_location = f'{data_root}/sheets/'
		self.num_samples = num_samples
		self.set_type = set_type
		self.set_list = np.load(f'{anno_root}/split.npy', allow_pickle='TRUE').item()

		names = [[f.split('/')[-1], f.split('/')[-2]] for f in glob((f'{self.img_location}/*/*'))]
		self.names = []
		self.classes_names = [[] for _ in classes]
		np.random.shuffle(names)
		for n in tqdm(names):
			if n[1] in classes:
				if os.path.exists(self.surf_location + n[1] + '/' + n[0] + '.npy'):
					if os.path.exists(self.touch_location + n[1] + '/' + n[0]):
						if n[0] + n[1] in self.set_list[self.set_type]:
							if n[0] +n[1] in self.set_list[self.set_type]:
								self.names.append(n)
								self.classes_names[classes.index(n[1])].append(n)

		print(f'The number of {set_type} set objects found : {len(self.names)}')

		self.pipeline = pipeline
		if self.pipeline is not None:
			self.pipeline = Compose(pipeline)

	def __len__(self):
		# return len(self.names)
		return 10

	# select the object and grasps for training
	def get_training_instance(self):
		# select an object and and a principle grasp randomly
		class_choice = random.choice(self.classes_names)
		object_choice = random.choice(class_choice)
		obj, obj_class = object_choice
		# select the remaining grasps and shuffle the select grasps
		num_choices = [0, 1, 2, 3, 4]
		nums = []
		for i in range(self.num_grasps):
			choice = random.choice(num_choices)
			nums.append(choice)
			del (num_choices[num_choices.index(choice)])
		random.shuffle(nums)
		return obj, obj_class, nums[-1], nums

	# select the object and grasps for validating
	def get_validation_examples(self, index):
		# select an object and a principle grasp
		obj, obj_class = self.names[index]
		orig_num = 0
		# select the remaining grasps deterministically
		nums = [(orig_num + i) % 5 for i in range(self.num_grasps)]
		return obj, obj_class, orig_num, nums

	# load surface point cloud
	def get_gt_points(self, obj_class, obj):
		samples = np.load(self.surf_location +obj_class + '/' + obj + '.npy')
		if self.test_mode == True:
			np.random.seed(0)
		np.random.shuffle(samples)
		gt_points = torch.FloatTensor(samples[:self.num_samples])
		gt_points *= .5 # scales the models to the size of shape we use
		gt_points[:, -1] += .6 # this is to make the hand and the shape the right releative sizes
		return gt_points

	# load vision signal
	def get_images(self, obj_class, obj, grasp_number):
		# load images
		img_occ = Image.open(f'{self.img_location}/{obj_class}/{obj}/{grasp_number}.png')
		img_unocc = Image.open(f'{self.img_location}/{obj_class}/{obj}/unoccluded.png')
		# apply pytorch image preprocessing
		img_occ = preprocess(img_occ)
		img_unocc = preprocess(img_unocc)
		return torch.FloatTensor(img_occ), torch.FloatTensor(img_unocc)

	# load touch sheet mask indicating toch success
	def get_touch_info(self, obj_class, obj, grasps):
		sheets, successful = [], []
		# cycle though grasps and load touch sheets
		for grasp in grasps:
			sheet_location = self.sheet_location + f'{obj_class}/{obj}/sheets_{grasp}_finger_num.npy'
			hand_info = np.load(f'{self.touch_location}/{obj_class}/{obj}/{grasp}.npy', allow_pickle=True).item()
			sheet, success = self.get_touch_sheets(sheet_location, hand_info)
			sheets.append(sheet)
			successful += success
		return torch.cat(sheets), successful

	# load the touch sheet
	def get_touch_sheets(self, location, hand_info):
		sheets = []
		successful = []
		touches = hand_info['touch_success']
		finger_pos = torch.FloatTensor(hand_info['cam_pos'])
		# cycle through fingers in the grasp
		for i in range(4):
			sheet = np.load(location.replace('finger_num', str(i)))
			# if the touch was unsuccessful
			if not touches[i] or sheet.shape[0] == 1:
				sheets.append(finger_pos[i].view(1, 3).expand(25, 3)) # save the finger position instead in every vertex
				successful.append(False) # binary mask for unsuccessful touch
			# if the touch was successful
			else:
				sheets.append(torch.FloatTensor(sheet)) # save the sheet
				successful.append(True) # binary mask for successful touch
		sheets = torch.stack(sheets)
		return sheets, successful

	def __getitem__(self, index):
		if self.set_type == 'train':
			obj, obj_class, grasp_number, grasps = self.get_training_instance()
		else:
			obj, obj_class, grasp_number, grasps = self.get_validation_examples(index)
		data = {}

		# meta data
		data['names'] = obj, obj_class, grasp_number
		data['class'] = obj_class

		# load sampled ground truth points
		data['gt_points'] = self.get_gt_points(obj_class, obj)

		# load images
		data['img_occ'], data['img_unocc'] = self.get_images(obj_class, obj, grasp_number)

		# get touch information
		data['sheets'], data['successful'] = self.get_touch_info(obj_class, obj, grasps)

		if self.pipeline is not None:
			data = self.pipeline(data)

		return data

	def evaluate(self,
				 results,
				 metric='loss',
				 **kwargs):
		assert metric == ['loss']
		total_loss = 0
		for losses in results:
			loss = losses['loss_dis']
			total_loss += loss.item()
		total_loss /= len(self)
		eval_res = {'loss': total_loss}
		return eval_res

# class used for obtaining an instance of the dataset for training touch chart prediction
# to be passed to a pytorch dataloader
# input:
#	- classes: list of object classes used
# 	- set_type: the set type used
# 	- num: if specified only returns a given grasp number
#	- all: if True use all objects, regarless of set type
#	- finger: if specified only returns a given finger number
@DATASETS.register_module()
class ABCTouchDataset(object):
	def __init__(self,
				 data_root,
				 anno_root,
				 pipeline=None,
				 classes=['0001', '0002'],
				 num_samples=4000,
				 set_type='train',
				 produce_sheets = False,
				 test_mode=False):

		assert set_type in ['train', 'valid', 'test']

		# initialization of data locations
		self.CLASSES = classes
		self.num_samples = num_samples
		self.surf_location = f'{data_root}/surfaces/'
		self.img_location = f'{data_root}/images/'
		self.touch_location = f'{data_root}/scene_info/'
		self.sheet_location = f'{data_root}/sheets/'
		self.set_type = set_type
		self.set_list = np.load(f'{anno_root}/split.npy', allow_pickle='TRUE').item()
		self.empty =  torch.FloatTensor(np.load(f'{anno_root}/empty_gel.npy'))
		self.produce_sheets = produce_sheets


		names = [[f.split('/')[-1], f.split('/')[-2]] for f in glob((f'{self.img_location}/*/*'))]
		self.names = []
		import os
		for n in tqdm(names):
			if n[1] in classes:
				if os.path.exists(self.surf_location + n[1]  + '/' + n[0] + '.npy'):
					if os.path.exists(self.touch_location + n[1] + '/' + n[0]):
						if self.produce_sheets or (n[0] + n[1]) in self.set_list[self.set_type]:
							if produce_sheets:
								for i in range(5):
									for j in range(4):
											self.names.append(n + [i, j])
							else:
								for i in range(5):
									hand_info = np.load(f'{self.touch_location}/{n[1]}/{n[0]}/{i}.npy',
														allow_pickle=True).item()
									for j in range(4):
										if hand_info['touch_success'][j]:
											self.names.append(n + [i, j])

		print(f'The number of {set_type} set objects found : {len(self.names)}')

		self.pipeline = pipeline
		if self.pipeline is not None:
			self.pipeline = Compose(pipeline)

	def __len__(self):
		return len(self.names)
		# return 10

	def standerdize_point_size(self, points):
		if points.shape[0] == 0:
			return torch.zeros((self.num_samples, 3))
		np.random.shuffle(points)
		points = torch.FloatTensor(points)
		while points.shape[0] < self.num_samples :
			points = torch.cat((points, points, points, points))
		perm = torch.randperm(points.shape[0])
		idx = perm[:self.num_samples ]
		return points[idx]

	def get_finger_transforms(self, hand_info, finger_num):
		rot = hand_info['cam_rot'][finger_num]
		rot = R.from_euler('xyz', rot, degrees=False).as_matrix()
		rot_q = R.from_matrix(rot).as_quat()
		pos = hand_info['cam_pos'][finger_num]
		return torch.FloatTensor(rot_q), torch.FloatTensor(rot), torch.FloatTensor(pos)


	def __getitem__(self, index):
		obj, obj_class, num, finger_num = self.names[index]

		# meta data
		data = {}
		data['names'] = [obj, num , finger_num]
		data['class'] = obj_class

		# hand infomation
		hand_info = np.load(f'{self.touch_location}/{obj_class}/{obj}/{num}.npy', allow_pickle=True).item()
		data['rot'], data['rot_M'], data['pos'] = self.get_finger_transforms(hand_info, finger_num)
		data['good_touch'] = hand_info['touch_success']

		# simulated touch information
		scene_info = np.load(f'{self.touch_location}/{obj_class}/{obj}/{num}.npy', allow_pickle=True).item()
		data['depth'] = torch.clamp(torch.FloatTensor(scene_info['depth'][finger_num]).unsqueeze(0), 0, 1)
		data['sim_touch']  = torch.FloatTensor(np.array(scene_info['gel'][finger_num]) / 255.).permute(2, 0, 1).contiguous().view(3, 100, 100)
		data['empty'] = torch.FloatTensor(self.empty / 255.).permute(2, 0, 1).contiguous().view(3, 100, 100)

		# point cloud information
		data['gt_points'] = self.standerdize_point_size(scene_info['points'][finger_num])
		data['num_samples'] = scene_info['points'][finger_num].shape

		# where to save sheets
		data['save_dir'] = f'{self.sheet_location}/{obj_class}/{obj}/sheets_{num}_{finger_num}.npy'

		if self.pipeline is not None:
			data = self.pipeline(data)

		return data

	def evaluate(self,
				 results,
				 metric='loss',
				 **kwargs):
		assert metric == ['loss']
		total_loss = 0
		for losses in results:
			loss = losses['loss_dis']
			total_loss += loss.item()
		total_loss /= len(self)
		eval_res = {'loss': total_loss}
		return eval_res


if __name__ == '__main__':
	dataset_touch = ABCTouchDataset(data_root='/hdd0/data/abc',
							  anno_root='/home/hanyang/rfvision/rfvision/models/detectors3d/touch_and_vision/data',
							  set_type='test')

	dataset_vision = ABCVisionDataset(data_root='/hdd0/data/abc',
							  anno_root='/home/hanyang/rfvision/rfvision/models/detectors3d/touch_and_vision/data',
							  set_type='train')