PARAMS_FILE_SRC=Parameters.py
PARAMS_FILE_BAK=Parameters.py.bak
LOGS_DIR=archives/logs
PATH="/cygdrive/c/Anaconda3":$PATH
export PATH

# for FILES_COUNT in 100 100 100 100 100 200 200 200 200 200 500 500 500 500 500 1000 1000 1000 1000 1000 1450 1450 1450 1450 1450; do
for FILES_COUNT in 1000 1000 1000 1000 1000 1450 1450 1450 1450 1450; do

  SUFFIX=`date +"%Y/%m/%d %H:%M:%S" | sed "s/[\/\:]//g" | sed "s/ /_/g"`
  
  echo "TRAINING_FILES_COUNT = ${FILES_COUNT}"
  cat $PARAMS_FILE_SRC > $PARAMS_FILE_BAK  
  REPLACEMENT="s/\: 100,     # if/\: ${FILES_COUNT},     # if/g"
  cat $PARAMS_FILE_SRC | sed "${REPLACEMENT}" > ${PARAMS_FILE_SRC}2
  mv ${PARAMS_FILE_SRC}2 $PARAMS_FILE_SRC

  ARCHIVE_DIR=archives/${SUFFIX}_FILES_COUNT_${FILES_COUNT}
  echo "ARCHIVE_DIR = ${ARCHIVE_DIR}"
  mkdir -p $ARCHIVE_DIR
  LOG_NAME=$ARCHIVE_DIR/log_${FILES_COUNT}_${SUFFIX}
  echo "LOG_NAME = $LOG_NAME"
  
  # log information about Parameters used
  echo "----------------------------------" >> $LOG_NAME
  echo "- Parameters.py                  -" >> $LOG_NAME
  echo "----------------------------------" >> $LOG_NAME
  cat $PARAMS_FILE_SRC >> $LOG_NAME
  
  # do training
  echo "Training"
  echo "----------------------------------" >> $LOG_NAME
  echo "- Training                       -" >> $LOG_NAME
  echo "----------------------------------" >> $LOG_NAME
  python Main.py --generate --train &>> $LOG_NAME
  
  # do testing
  # echo "Testing"
  # echo "----------------------------------" >> $LOG_NAME
  # echo "- Testing                        -" >> $LOG_NAME
  # echo "----------------------------------" >> $LOG_NAME
  # python Main.py --segment --count &>> $LOG_NAME 
  
  # archive results
  TRAIN_DIR=`ls -1 data | grep train_`
  echo $TRAIN_DIR
  
  echo "Moving results to archive dir"
  cd data
  mv $TRAIN_DIR ../$ARCHIVE_DIR
  # mv segmented ../$ARCHIVE_DIR
  cd ..

  # recover old Parameters.py file
  mv $PARAMS_FILE_BAK $PARAMS_FILE_SRC
done