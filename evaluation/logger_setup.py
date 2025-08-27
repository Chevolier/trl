import logging

def setup_logging():
    logging.basicConfig(
        level="INFO",
        format='%(asctime)s %(process)s %(levelname)s(%(module)s:%(lineno)d) - %(message)s',
        handlers=[
            logging.StreamHandler(),
        ]
    )
    return logging.getLogger(__name__)
