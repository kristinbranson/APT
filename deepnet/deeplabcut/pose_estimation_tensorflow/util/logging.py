"""
Modified by Mayank Kabra
Adapted from DeepLabCut2.0 Toolbox (deeplabcut.org)
Copyright A. & M. Mathis Labs
https://github.com/AlexEMG/DeepLabCut

Please see AUTHORS for contributors.
https://github.com/AlexEMG/DeepLabCut/blob/master/AUTHORS
Licensed under GNU Lesser General Public License v3.0

Adapted from DeeperCut by Eldar Insafutdinov
https://github.com/eldar/pose-tensorflow

"""
import logging, os


def setup_logging():
    FORMAT = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename='log.txt', filemode='w',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO, format=FORMAT)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)