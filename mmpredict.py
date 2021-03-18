import sys
# sys.path.extend(['/home/palm/PycharmProjects/mmdetection'])
from mmdet.apis import inference_detector, init_detector, show_result_pyplot
from mmcv import Config
import cv2
import json
import os


if __name__ == '__main__':
    dataset = json.load(open('/home/palm/PycharmProjects/mmdetection/anns/test.json'))

    cfg = Config.fromfile('configs/cascade_rcnn/cascade_rcnn_r101_fpn_1x_algea.py')

    # Build the detector
    model = init_detector(cfg, '/home/palm/PycharmProjects/algea3/cascade_101_lab_1/epoch_20.pth', device='cpu')
    os.makedirs('/media/palm/BiggerData/algea/cascade_101_lab_1', exist_ok=True)
    for data in dataset:
        # test a single image
        result = inference_detector(model, data['filename'])
        # show the results
        img = model.show_result(data['filename'],
                                result,
                                score_thr=0.3, show=False)
        print(result)
        # cv2.imwrite(os.path.join('/media/palm/BiggerData/algea/cascade_101_lab_1',
        #                          os.path.basename(data['filename'])),
        #             img)
    # show_result_pyplot(model, '/media/palm/data/MicroAlgae/22_11_2020/images/00001.jpg', result, score_thr=0.3)
