import os
import math

def create_dir(path):
	if not os.path.exists(path):
		os.mkdir(path)

def lr_schedule_cosdecay(t, T, init_lr=0.0001):
	lr = 0.5 * (1 + math.cos(t * math.pi / T)) * init_lr
	return lr




