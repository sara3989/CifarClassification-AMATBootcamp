from dataset_preparations import dataset_preparations
import logging

if __name__ == '__main__':
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    dataset_preparations.dataset_preparations()
    logging.info('Data is prepared now to modeling')
