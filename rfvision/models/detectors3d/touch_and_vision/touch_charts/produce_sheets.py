#Copyright (c) Facebook, Inc. and its affiliates.
#All rights reserved.

#This source code is licensed under the license found in the
#LICENSE file in the root directory of this source tree.

import os
import torch
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import sys
sys.path.insert(0, "../")
from rfvision.models.detectors3d.touch_and_vision.abc_dataset import ABCTouchDataset
from rfvision.models.detectors3d.touch_and_vision.touch_charts.models import TouchEncoder
from rfvision.models.detectors3d.touch_and_vision.utils import load_mesh_touch, chamfer_distance, batch_calc_edge

data_root = '/hdd0/data/abc/'
anno_root = '/home/hanyang/rfvision/rfvision/models/detectors3d/touch_and_vision/data'


class Engine():
	def __init__(self, args):

		# set seeds
		np.random.seed(args.seed)
		torch.manual_seed(args.seed)
		torch.cuda.manual_seed(args.seed)

		self.classes = ['0001', '0002']
		self.verts, self.faces = load_mesh_touch(f'../data/initial_sheet.obj')
		self.args = args
	def __call__(self) -> float:
		self.encoder = TouchEncoder()
		self.encoder.load_state_dict(torch.load(self.args.save_directory))
		self.encoder.cuda()
		self.encoder.eval()

		train_data = ABCTouchDataset(data_root, anno_root, classes=self.classes, produce_sheets=True)
		train_data.names = train_data.names[self.args.start:self.args.end]
		train_loader = DataLoader(train_data, batch_size=1, shuffle=False,
					num_workers=16, collate_fn=train_data.collate)

		for k, batch in enumerate(tqdm(train_loader, smoothing=0)):
			# initialize data
			sim_touch = batch['sim_touch'].cuda()
			depth = batch['depth'].cuda()
			ref_frame = batch['ref']

			# predict point cloud
			with torch.no_grad():
				pred_depth, sampled_points = self.encoder(sim_touch, depth, ref_frame, empty = batch['empty'].cuda())

			# optimize touch chart
			for points, dir in zip(sampled_points, batch['save_dir']):
				if os.path.exists(dir):
					continue
				directory = dir[:-len(dir.split('/')[-1])]
				if not os.path.exists(directory):
					os.makedirs(directory)

				# if not a successful touch
				if torch.abs(points).sum() == 0 :
					np.save(dir, np.zeros(1))
					continue

				# make initial mesh match touch sensor when touch occurred
				initial = self.verts.clone().unsqueeze(0)
				pos = ref_frame['pos'].cuda().view(1, -1)
				rot = ref_frame['rot_M'].cuda().view(1, 3, 3)
				initial = torch.bmm(rot, initial.permute(0, 2, 1)).permute(0, 2, 1)
				initial += pos.view(1, 1, 3)
				initial = initial[0]

				# set up optimization
				updates = torch.zeros(self.verts.shape, requires_grad=True, device="cuda")
				optimizer = optim.Adam([updates], lr=0.003, weight_decay=0)
				last_improvement = 0
				best_loss = 10000

				while True:
					# update
					optimizer.zero_grad()
					verts = initial + updates

					# losses
					surf_loss = chamfer_distance(verts.unsqueeze(0), self.faces, points.unsqueeze(0), num =self.args.num_samples)
					edge_lengths = batch_calc_edge(verts.unsqueeze(0), self.faces)
					loss = self.args.surf_co * surf_loss + 70 * edge_lengths

					# optimize
					loss.backward()
					optimizer.step()

					# check results
					if loss < 0.0006:
						break
					if best_loss > loss :
						best_loss = loss
						best_verts = verts.clone()
						last_improvement = 0
					else:
						last_improvement += 1
						if last_improvement > 50:
							break

				np.save(dir, best_verts.data.cpu().numpy())



if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--seed', type=int, default=0, help='Random seed.')
	parser.add_argument('--start', type=int, default=0, help='Random seed.')
	parser.add_argument('--end', type=int, default=10000000, help='Random seed.')
	parser.add_argument('--save_directory', type=str, default='experiments/checkpoint/pretrained/encoder_touch',
						help='Location of the model used to produce sheet')
	parser.add_argument('--num_samples', type=int, default=4000, help='Number of points in the predicted point cloud.')
	parser.add_argument('--model_location', type=str, default="../data/initial_sheet.obj",
						help='Location of inital mesh sheet whcih will be optimized')
	parser.add_argument('--surf_co', type=float, default=9000.)
	args = parser.parse_args()
	trainer = Engine(args)
	trainer()







