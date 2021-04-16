import torch
import tensorrt as trt


if trt.__version__ >= '5.1':
    DEFAULT_CALIBRATION_ALGORITHM = trt.CalibrationAlgoType.ENTROPY_CALIBRATION_2
else:
    DEFAULT_CALIBRATION_ALGORITHM = trt.CalibrationAlgoType.ENTROPY_CALIBRATION
        
    
class DatasetCalibrator(trt.IInt8Calibrator):
    
    def __init__(self, dataloader, algorithm=DEFAULT_CALIBRATION_ALGORITHM):
        super(DatasetCalibrator, self).__init__()
        
        self.batch_size = dataloader.batch_size
        self.dataloader = iter(dataloader)
        self.algorithm  = algorithm
        
    def get_batch(self, *args, **kwargs):
        try:
            inputs = next(self.dataloader)
            return [int(i.data_ptr()) for i in inputs]
        except StopIteration:
            return []
        
    def get_algorithm(self):
        return self.algorithm
    
    def get_batch_size(self):
        return self.batch_size
    
    def read_calibration_cache(self, *args, **kwargs):
        return None
    
    def write_calibration_cache(self, cache, *args, **kwargs):
        pass