import os
import time
import torch
import pickle
import datetime
import numpy as np
from tqdm import tqdm
import os.path as osp
import torch.optim as optim
# from utils.nms_wrapper import nms
from torchvision.ops import nms
from torch.cuda.amp import GradScaler, autocast
from utils.genutils import to_var, write_print, mkdir

from utils.timer import Timer
from loss.loss import get_loss
from models.model import get_model
from layers.anchor_box import AnchorBox
from data.coco import save_results as coco_save
from data.pascal_voc import save_results as voc_save
from data.pascal_voc import do_python_eval as do_voc_eval
from data.mmmdad import save_results as mmmdad_save
from data.mmmdad import do_python_eval as do_mmmdad_eval
from pycocotools.cocoeval import COCOeval as do_coco_eval


class Solver(object):

    DEFAULTS = {}

    def __init__(self,
                 version,
                 data_loader,
                 config,
                 output_txt):
        """
        Initializes a Solver object
        """

        super(Solver, self).__init__()
        self.__dict__.update(Solver.DEFAULTS, **config)
        self.version = version
        self.data_loader = data_loader
        self.config = config
        self.output_txt = output_txt

        self.build_model()

        # start with a pre-trained model
        if self.pretrained_model:
            self.load_pretrained_model()
        else:
            self.model.init_weights(self.model_save_path,
                                    self.basenet)

    def build_model(self):
        """
        Instantiate the model, loss criterion, and optimizer
        """

        # instantiate anchor boxes
        anchor_boxes = AnchorBox(new_size=self.new_size,
                                 config=self.anchor_config,
                                 scale_initial=self.scale_initial,
                                 scale_min=self.scale_min)
        self.anchor_boxes = anchor_boxes.get_boxes()

        if torch.cuda.is_available() and self.use_gpu:
            self.anchor_boxes = self.anchor_boxes.cuda()

        # instatiate model
        self.model = get_model(config=self.config,
                               anchors=self.anchor_boxes)

        # instatiate loss criterion
        self.criterion = get_loss(config=self.config)

        # instatiate optimizer
        self.optimizer = optim.SGD(params=self.model.parameters(),
                                   lr=self.lr,
                                   momentum=self.momentum,
                                   weight_decay=self.weight_decay)

        self.scaler = GradScaler()

        # print network
        self.print_network(self.model)

        # use gpu if enabled
        if torch.cuda.is_available() and self.use_gpu:
            self.model.cuda()
            self.criterion.cuda()

    def print_network(self, model):
        """
        Prints the structure of the network and the total number of parameters
        """
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        write_print(self.output_txt, str(model))
        write_print(self.output_txt,
                    'The number of parameters: {}'.format(num_params))

    def load_pretrained_model(self):
        """
        loads a pre-trained model from a .pth file
        """
        self.model.load_state_dict(torch.load(os.path.join(
            self.model_save_path, '{}.pth'.format(self.pretrained_model))))
        write_print(self.output_txt,
                    'loaded trained model {}'.format(self.pretrained_model))

    def adjust_learning_rate(self,
                             optimizer,
                             gamma,
                             step):
        """Sets the learning rate to the initial LR decayed by 10 at every
            specified step
        # Adapted from PyTorch Imagenet example:
        # https://github.com/pytorch/examples/blob/master/imagenet/main.py
        """
        lr = self.lr * (gamma ** (step))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def print_loss_log(self,
                       start_time,
                       iters_per_epoch,
                       e,
                       i,
                       class_loss,
                       loc_loss,
                       loss):
        """
        Prints the loss and elapsed time for each epoch
        """

        """
        Prints the loss and elapsed time for each epoch
        """
        total_iter = self.num_epochs * iters_per_epoch
        cur_iter = e * iters_per_epoch + i

        elapsed = time.time() - start_time
        total_time = (total_iter - cur_iter) * elapsed / (cur_iter + 1)
        epoch_time = (iters_per_epoch - i) * elapsed / (cur_iter + 1)

        epoch_time = str(datetime.timedelta(seconds=epoch_time))
        total_time = str(datetime.timedelta(seconds=total_time))
        elapsed = str(datetime.timedelta(seconds=elapsed))

        log = "Elapsed {}/{} -- {}, Epoch [{}/{}], Iter [{}/{}], " \
              "class_loss: {:.4f}, loc_loss: {:.4f}, " \
              "loss: {:.4f}".format(elapsed,
                                    epoch_time,
                                    total_time,
                                    e + 1,
                                    self.num_epochs,
                                    i + 1,
                                    iters_per_epoch,
                                    class_loss.item(),
                                    loc_loss.item(),
                                    loss.item())

        write_print(self.output_txt, log)

    def save_model(self, e):
        """
        Saves a model per e epoch
        """
        path = os.path.join(
            self.model_save_path,
            '{}/{}.pth'.format(self.version, e + 1)
        )

        torch.save(self.model.state_dict(), path)


    # def model_step(self,
    #                images,
    #                targets,
    #                count):
        """
        A step for each iteration
        """

        """
        if count == 0:
            # update parameters
            # self.optimizer.step()
            self.scaler.step(self.optimizer)

            # empty the gradients of the model through the optimizer
            self.optimizer.zero_grad()

            self.scaler.update()

            count = self.batch_multiplier

        with autocast():

            # forward pass
            class_preds, loc_preds = self.model(images)

            # compute loss
            class_targets = [target[:, -1] for target in targets]
            loc_targets = [target[:, :-1] for target in targets]
            losses = self.criterion(class_preds=class_preds,
                                    class_targets=class_targets,
                                    loc_preds=loc_preds,
                                    loc_targets=loc_targets,
                                    anchors=self.anchor_boxes)

            class_loss, loc_loss, loss = losses

        # compute gradients using back propagation
        # loss.backward()
        class_loss = class_loss / self.batch_multiplier
        loc_loss = loc_loss / self.batch_multiplier
        loss = loss / self.batch_multiplier
        self.scaler.scale(loss).backward()
        count = count - 1

        # return loss
        return class_loss, loc_loss, loss, count
    """

    def model_step(self,
                   images,
                   targets,
                   count):
        """
        # A step for each iteration
        """

        # empty the gradients of the model through the optimizer
        self.optimizer.zero_grad()

        with autocast():

            # forward pass
            class_preds, loc_preds = self.model(images)

            # compute loss
            class_targets = [target[:, -1] for target in targets]
            loc_targets = [target[:, :-1] for target in targets]
            losses = self.criterion(class_preds=class_preds,
                                    class_targets=class_targets,
                                    loc_preds=loc_preds,
                                    loc_targets=loc_targets,
                                    anchors=self.anchor_boxes)

            class_loss, loc_loss, loss = losses

        # compute gradients using back propagation
        # loss.backward()
        self.scaler.scale(loss).backward()

        # update parameters
        # self.optimizer.step()
        self.scaler.step(self.optimizer)

        self.scaler.update()

        # return loss
        return class_loss, loc_loss, loss, count

    def train(self):
        """
        training process
        """

        # set model in training mode
        self.model.train()

        self.losses = []
        count = self.batch_multiplier

        iters_per_epoch = len(self.data_loader)

        # start with a trained model if exists
        if self.pretrained_model:
            start = int(self.pretrained_model.split('/')[-1])
        else:
            start = 0

        sched = 0

        # start training
        start_time = time.time()
        for e in range(start, self.num_epochs):
            for i, (images, targets) in enumerate(tqdm(self.data_loader)):
                images = to_var(images, self.use_gpu)
                targets = [to_var(target, self.use_gpu) for target in targets]

                class_loss, loc_loss, loss, count = self.model_step(images,
                                                                    targets,
                                                                    count)

                # print out loss log
                # self.print_loss_log(start_time=start_time,
                #                     iters_per_epoch=iters_per_epoch,
                #                     e=e,
                #                     i=i,
                #                     class_loss=class_loss,
                #                     loc_loss=loc_loss,
                #                     loss=loss)
            if (e + 1) % self.loss_log_step == 0:
                self.print_loss_log(start_time=start_time,
                                    iters_per_epoch=iters_per_epoch,
                                    e=e,
                                    i=i,
                                    class_loss=class_loss,
                                    loc_loss=loc_loss,
                                    loss=loss)

                self.losses.append([e, class_loss, loc_loss, loss])

            # save model
            if (e + 1) % self.model_save_step == 0:
                self.save_model(e)

            num_sched = len(self.learning_sched)
            if num_sched != 0 and sched < num_sched:
                if (e + 1) == self.learning_sched[sched]:

                    self.lr /= 10
                    write_print(self.output_txt,
                                'Learning rate reduced to ' + str(self.lr))
                    sched += 1
                    self.adjust_learning_rate(optimizer=self.optimizer,
                                              gamma=self.sched_gamma,
                                              step=sched)

        # print losses
        write_print(self.output_txt, '\n--Losses--')
        for e, class_loss, loc_loss, loss in self.losses:
            loss_string = ' {:.4f} {:.4f} {:.4f}'.format(class_loss,
                                                         loc_loss,
                                                         loss)
            write_print(self.output_txt, str(e) + loss_string)

    def eval(self,
             dataset,
             max_per_image,
             score_threshold):

        num_images = len(dataset)
        all_boxes = [[[] for _ in range(num_images)]
                     for _ in range(self.class_count)]

        # prepare timers, paths, and files
        timer = {'detection': Timer(), 'nms': Timer()}

        results_path = osp.join(self.model_test_path,
                                self.pretrained_model)
        mkdir(results_path)
        detection_file = osp.join(results_path,
                                  'detections.pkl')

        detect_times = []
        nms_times = []

        with torch.no_grad():

            # for each image
            for i in range(num_images):

                # get image
                image, target, h, w = dataset.pull_item(i)
                image = to_var(image.unsqueeze(0), self.use_gpu)

                # get and time detection
                timer['detection'].tic()
                bboxes, scores = self.model(image)
                detect_time = timer['detection'].toc(average=False)
                detect_times.append(detect_time)

                # convert to CPU tensors
                bboxes = bboxes[0]
                scores = scores[0]
                # bboxes = bboxes.cpu().numpy()
                # scores = scores.cpu().numpy()

                # scale each detection back up to the image
                # scale = torch.Tensor([w, h, w, h]).cpu().numpy()
                scale = torch.Tensor([w, h, w, h])
                bboxes *= scale

                # perform and time NMS
                timer['nms'].tic()

                for j in range(1, self.class_count):

                    # get scores greater than score_threshold
                    selected_i = np.where(scores[:, j] > score_threshold)[0]

                    # if there are scores greater than score_threshold
                    if len(selected_i) > 0:
                        bboxes_i = bboxes[selected_i]
                        scores_i = scores[selected_i, j]
                        detections_i = (bboxes_i, scores_i[:, np.newaxis])
                        detections_i = np.hstack(detections_i)
                        # detections_i = detections_i.astype(np.float32,
                        #                                    copy=False)

                        # keep = nms(detections=detections_i,
                        #            threshold=self.nms_threshold,
                        #            force_cpu=True)
                        # print(type(bboxes_i), type(scores_io: object))
                        keep = nms(boxes=bboxes_i,
                                   scores=scores_i,
                                   iou_threshold=self.nms_threshold)

                        keep = keep[:50]
                        detections_i = detections_i[keep, :]
                        if len(detections_i.shape) == 1:
                            all_boxes[j][i] = np.expand_dims(detections_i, 0)
                        else:
                            all_boxes[j][i] = detections_i

                    elif len(selected_i) == 0:
                        all_boxes[j][i] = np.empty([0, 5], dtype=np.float32)

                # if we need to limit the maximum per image
                if max_per_image > 0:

                    # get all the scores for the image across all classes
                    scores_i = np.hstack([all_boxes[j][i][:, -1]
                                          for j in range(1, self.class_count)])

                    # if the number of detections is greater than max_per_image
                    if len(scores_i) > max_per_image:

                        # get the score of the max_per_image-th image
                        threshold_i = np.sort(scores_i)[-max_per_image]

                        # keep detections with score greater than threshold_i
                        for j in range(1, self.class_count):
                            keep = np.where(all_boxes[j][i][:, -1]
                                            >= threshold_i)[0]
                            all_boxes[j][i] = all_boxes[j][i][keep, :]

                nms_time = timer['nms'].toc(average=False)
                nms_times.append(nms_time)

                temp_string = 'detection: {:d}/{:d} {:.4f}s {:.4f}s'
                temp_string = temp_string.format(i + 1,
                                                 num_images,
                                                 detect_time,
                                                 nms_time)

                write_print(self.output_txt, temp_string)

        with open(detection_file, 'wb') as f:
            pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

        write_print(self.output_txt, '\nEvaluating detections')

        # perform evaluation
        if self.dataset == 'voc':
            voc_save(all_boxes=all_boxes,
                     dataset=dataset,
                     results_path=results_path,
                     output_txt=self.output_txt)

            aps, mAP = do_voc_eval(results_path=results_path,
                                   dataset=dataset,
                                   output_txt=self.output_txt,
                                   mode='test',
                                   iou_threshold=self.iou_threshold,
                                   use_07_metric=self.use_07_metric)

            write_print(self.output_txt, '\nResults:')
            for ap in aps:
                write_print(self.output_txt, '{:.4f}'.format(ap))
            write_print(self.output_txt, '{:.4f}'.format(np.mean(aps)))

        if self.dataset == 'mmmdad':
            mmmdad_save(all_boxes=all_boxes,
                     dataset=dataset,
                     results_path=results_path,
                     output_txt=self.output_txt)

            aps, mAP = do_mmmdad_eval(results_path=results_path,
                                   dataset=dataset,
                                   output_txt=self.output_txt,
                                   iou_threshold=self.iou_threshold,
                                   use_07_metric=self.use_07_metric)

            write_print(self.output_txt, '\nResults:')
            for ap in aps:
                write_print(self.output_txt, '{:.4f}'.format(ap))
            write_print(self.output_txt, '{:.4f}'.format(np.mean(aps)))

        if self.dataset == 'coco':
            detection_list = coco_save(all_boxes=all_boxes,
                                       dataset=dataset,
                                       results_path=results_path,
                                       output_txt=self.output_txt)

            detection_list = dataset.pycoco.loadRes(detection_list)
            coco_eval = do_coco_eval(dataset.pycoco,
                                     detection_list,
                                     'bbox')
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()

            stats = ['AP--IoU=0.50:0.95--all--100',
                     'AP--IoU=0.50--all--100',
                     'AP--IoU=0.75--all-100',
                     'AP--IoU=0.50:0.95--small--100',
                     'AP--IoU=0.50:0.95--medium--100',
                     'AP--IoU=0.50:0.95--large--100',
                     'AR--IoU=0.50:0.95--all--1',
                     'AR--IoU=0.50:0.95--all--10',
                     'AR--IoU=0.50:0.95--all--100',
                     'AR--IoU=0.50:0.95--small--100',
                     'AR--IoU=0.50:0.95--medium--100',
                     'AR--IoU=0.50:0.95--large--100']

            for stat, val in zip(stats, coco_eval.stats):
                str_out = '{:s}: {:.3f}'.format(stat, val)
                write_print(self.output_txt, str_out)

            write_print(self.output_txt, '\nResults:')
            for val in coco_eval.stats:
                write_print(self.output_txt, '{:.3f}'.format(val))

        detect_times = np.asarray(detect_times)
        nms_times = np.asarray(nms_times)
        total_times = np.add(detect_times, nms_times)
        write_print(self.output_txt, str(1 / np.mean(detect_times[1:])))
        write_print(self.output_txt, str(1 / np.mean(nms_times[1:])))
        write_print(self.output_txt, str(1 / np.mean(total_times[1:])))

    def test(self):
        """
        testing process
        """
        self.model.eval()
        self.eval(dataset=self.data_loader.dataset,
                  max_per_image=self.max_per_image,
                  score_threshold=self.score_threshold)
