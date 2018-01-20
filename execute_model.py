import argparse
import matplotlib.pyplot as plot

import chainer

from chainercv.links import SSD300
from chainercv import utils
from chainercv.visualizations import vis_bbox

LABEL_NAMES = ('other',
               'berlinerdom',
               'brandenburgertor',
               'fernsehturm',
               'funkturm',
               'reichstag',
               'rotesrathaus',
               'siegessaeule',
               'none')


class Predictor():

    def __init__(self, trained_model, gpu=-1):
        try:
            self.model = SSD300(
                n_fg_class=len(LABEL_NAMES),
                pretrained_model=trained_model)

            if gpu >= 0:
                chainer.cuda.get_device_from_id(gpu).use()
                self.model.to_gpu()
        except:
            print('Could not load model')

    def run(self, image, vis=False):
        if hasattr(self, 'model'):
            img = utils.read_image(image, color=True)
            bboxes, labels, scores = self.model.predict([img])

            if vis:
                [vis_bbox(img, bbox, label, score, label_names=LABEL_NAMES)
                 for bbox, label, score in zip(bboxes, labels, scores)]
                plot.show()
            bboxes = list(map((lambda x: x.tolist()), bboxes))
            labels = list(map((lambda x: x.tolist()), labels))
            scores = list(map((lambda x: x.tolist()), scores))
            return bboxes, labels, scores
        print('Model was not correctly loaded, can\'t predict anything!')
        return [], [], []


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Running configured \
                                                  multibox detector')
    parser.add_argument('--model', default='../result/model')
    parser.add_argument('--gpu', default=-1)
    parser.add_argument('--vis', dest='vis', action='store_true')
    parser.set_defaults(vis=False)
    parser.add_argument('image')
    args = parser.parse_args()
    Predictor(args.model, args.gpu).run(args.image, args.vis)
