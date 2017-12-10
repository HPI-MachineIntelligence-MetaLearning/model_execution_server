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


def run(image, trained_model, gpu=-1, vis=False):
    try:
        model = SSD300(
            n_fg_class=len(LABEL_NAMES),
            pretrained_model=trained_model)

        if gpu >= 0:
            chainer.cuda.get_device_from_id(gpu).use()
            model.to_gpu()

        img = utils.read_image(image, color=True)
        bboxes, labels, scores = model.predict([img])

        if vis:
            [vis_bbox(img, bbox, label, score, label_names=LABEL_NAMES)
             for bbox, label, score in zip(bboxes, labels, scores)]
            plot.show()
        bboxes = list(map((lambda x: x.tolist()), bboxes))
        labels = list(map((lambda x: x.tolist()), labels))
        scores = list(map((lambda x: x.tolist()), scores))
        return bboxes, labels, scores
    except:
        print('Could not load model')
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
    run(args.image, args.model, args.gpu, args.vis)
