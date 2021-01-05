from lib.helper.logger import logger
from lib.core.base_trainer.net_work import trainner
import setproctitle

import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

logger.info('train start')
setproctitle.setproctitle("detect")

trainner=trainner()

trainner.train()
