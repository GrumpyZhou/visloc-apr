from utils.common.setup import config_to_string
from utils.common.visdom_manager import *
import datetime

class OptimSearchVisTmp:   
    def get_meters(config):
        if config.visenv is None:
            visman = DummyVisManager()
            loss_meter = DummyVisObj()
            pos_acc_meter = DummyVisObj()
            rot_acc_meter = DummyVisObj()
        else:
            # Setup visualizer
            if config.viswin is not None:
                loss_win_tag = '{}-Loss'.format(config.viswin)
                acc_win_tag = '{}-Val'.format(config.viswin)
            else:
                loss_win_tag = 'Loss'
                acc_win_tag = 'Val'
            legend = config.optim_tag
            pos_legend = 'pos_{}'.format(legend)
            rot_legend = 'rot_{}'.format(legend)
            targets={loss_win_tag : [legend], acc_win_tag : [pos_legend, rot_legend]}
            visman = VisManager(config.visenv, config.viswin, targets, host=config.vishost, port=config.visport, enable_log=False)
            now = datetime.datetime.now()
            #visman.log('>>>Log window for {}<br>{}/{}/{} {}:{}:{}<br>'.format(config.viswin, now.day, now.month, now.year, now.hour, now.minute, now.second))
            #visman.log(config_to_string(config, html=True))
            loss_meter = visman.get_win(loss_win_tag, legend)
            pos_acc_meter = visman.get_win(acc_win_tag, pos_legend)
            rot_acc_meter = visman.get_win(acc_win_tag, rot_legend)

            # Set states
            if config.start_epoch >= 1:
                loss_meter.inserted = True
            if config.start_epoch >= config.validate:
                pos_acc_meter.inserted = True
                rot_acc_meter.inserted = True     
        return visman, loss_meter, pos_acc_meter, rot_acc_meter

class PoseNetVisTmp:   
    def get_meters(config, with_losses=False, with_homos=False):
        if config.visenv is None:
            visman = DummyVisManager()
            tloss_meter = DummyVisObj()
            losses_meters = [(DummyVisObj(), DummyVisObj()) for i in range(3)]
            homo_meters = [DummyVisObj(), DummyVisObj()]
            pos_acc_meter = DummyVisObj()
            rot_acc_meter = DummyVisObj()
        else:
            # Setup visualizer
            if config.viswin is not None:
                acc_win_tag = '{}-Val'.format(config.viswin)
                loss_win_tag = '{}-Losses'.format(config.viswin)
                homo_win_tag = '{}-Homos'.format(config.viswin)
            else:
                acc_win_tag = '{}-Loss-Val'.format(config.dataset)
                loss_win_tag = '{}-Losses'.format(config.dataset)
                homo_win_tag = '{}-Homos'.format(config.viswin)
            loss_win_legends = []
            for i in range(3):
                loss_win_legends.append('l{}_x'.format(i))
                loss_win_legends.append('l{}_q'.format(i))
            homo_win_legends = ['sx', 'sq']                
            tloss_legend = 'loss'
            pos_legend = 'pos'
            rot_legend = 'rot'
            targets = {acc_win_tag : [tloss_legend, pos_legend, rot_legend]}
            if with_losses:
                targets.update({loss_win_tag : loss_win_legends})
            if with_homos:
                targets.update({homo_win_tag : homo_win_legends})
            visman = VisManager(config.visenv, config.viswin, targets, host=config.vishost, port=config.visport)
            now = datetime.datetime.now()
            
            # Meters for losses
            losses_meters = []
            if with_losses:
                for i in range(3):
                    losses_meters.append((visman.get_win(loss_win_tag, 'l{}_x'.format(i)),
                                          visman.get_win(loss_win_tag, 'l{}_q'.format(i))))
                                          
            # Meters for homoscedastic uncertainties
            homo_meters = []
            if with_homos:
                homo_meters = [visman.get_win(homo_win_tag, 'sx'), visman.get_win(homo_win_tag, 'sq')]
                         
            # Meters for loss and val
            tloss_meter = visman.get_win(acc_win_tag, tloss_legend)
            pos_acc_meter = visman.get_win(acc_win_tag, pos_legend)
            rot_acc_meter = visman.get_win(acc_win_tag, rot_legend)

            # Set states
            if config.start_epoch >= 1:
                tloss_meter.inserted = True
                homo_meters[0].inserted = True
                homo_meters[1].inserted = True
                for i, val in enumerate(losses_meters):
                    losses_meters[i][0].inserted = True
                    losses_meters[i][1].inserted = True
            if config.start_epoch >= config.validate:
                pos_acc_meter.inserted = True
                rot_acc_meter.inserted = True            
        return visman, tloss_meter, pos_acc_meter, rot_acc_meter, losses_meters, homo_meters
