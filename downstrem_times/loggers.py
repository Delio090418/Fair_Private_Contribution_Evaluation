import logging
import os

##loggers several quantities


def logger_f(message,log_path):       
    log_dir = os.path.dirname(log_path)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logger = logging.getLogger(log_path)

    if not logger.hasHandlers():  # Prevent duplicate handlers
        logger.setLevel(logging.INFO)

        app_handler = logging.FileHandler(log_path)
        app_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        app_handler.setFormatter(app_formatter)

        logger.addHandler(app_handler)

    logger.info(message)  
   
if __name__=="__main__":
    a=2+1
    message=f"test: {a}"
    name_1="metrics1"
    name_2="metrics2"
    logger_f(message,name_1)
    logger_f(message,name_2)