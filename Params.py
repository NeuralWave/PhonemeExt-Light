
class default:
	NMELCHANNELS = 80
	SR = 22050
	WINDOW = 'hann'
	STFT_SAMPLES = 1024
	STFT_STEPSZ = 256
	FMIN = 125
	FMAX = 7600
	NPHNCLASSES = 62

class normalization_modes:
	DEFAULT = "MAX_DIVIDE"

	NONE = "NONE"
	NORMALIZE = "NORMALIZE"
	NORMALIZE_POSITIVE = "NORMALIZE_POSITIVE"
	MAX_DIVIDE = "MAX_DIVIDE"

class train_params:
	BZ = 16
	SHUFFLE = True
	NUM_WORKERS = 0
	LR = 1e-4
	WGT_DECAY = 1e-4
	GRADIENT_CLIP=False
	USE_TENSOR_CORES=False # set sizes to multiples of 8 if true

class model_params:
	###### Conv layers ######
	N_CONV_LAYERS_1 = 4
	N_CONV_LAYERS_2 = 6

	KERNEL_CONV_1 = 5
	KERNEL_CONV_2 = 5

	CONV_FEAT_1 = 128
	CONV_FEAT_2 = 256
	CONV_FEAT_OUT = 256

	KERNEL_POOL_1 = 5
	KERNEL_POOL_2 = 1

	DROPOUT_CONV = 0.01	
	BATCHNORM_MOMENTUM = 0.1
	BATCHNORM_RUN_STATS = True
	########################

	###### Lin layers ######
	N_LIN_LAYERS = 3
	LIN_DIM = 1024
	DROPOUT_LIN = 0.3
	########################