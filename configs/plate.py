alphabet = "京沪津渝冀晋蒙辽吉黑苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新学警港澳挂使领民航应急0123456789ABCDEFGHJKLMNPQRSTUVWXYZO"

# alphabet = "京沪津渝冀晋蒙辽吉黑苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新学警港澳挂使领民航应急0123456789ABCDEFGHJKLMNPQRSTUVWXYZOI"

K = 8
adam = True
adadelta = False
saveInterval = 1
valInterval = 200
lr_step = 25  # default is 50
n_test_disp = 10
displayInterval = 500
gpu = 1

experiment = "output_1"
model_type = "CNN"  #

train_data = "../clpr_lmdb/train"
val_data = "../clpr_lmdb/val"

pre_model = ""
beta1 = 0.5
lr = 0.001
niter = 150  # default is 300

imgW = 96
imgH = 32
nc = 1
val_batchSize = 256
batchSize = 256

workers = 16
