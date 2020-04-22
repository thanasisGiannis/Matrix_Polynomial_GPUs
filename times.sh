#!/bin/bash

POLDEG=5

for((MATRIX=1000;MATRIX<=2000;MATRIX=MATRIX+1))  
   do
        ./test -r $MATRIX -d $POLDEG  >> test1.csv
		sleep 1
		./test -r $MATRIX -d $POLDEG  >> test1.csv
		sleep 1
		./test -r $MATRIX -d $POLDEG  >> test1.csv
		sleep 1
		./test -r $MATRIX -d $POLDEG  >> test1.csv
		sleep 1
		./test -r $MATRIX -d $POLDEG  >> test1.csv
		sleep 1
		./test -r $MATRIX -d $POLDEG  >> test1.csv
		sleep 1
		./test -r $MATRIX -d $POLDEG  >> test1.csv
		sleep 1
		./test -r $MATRIX -d $POLDEG  >> test1.csv
		sleep 1
		./test -r $MATRIX -d $POLDEG  >> test1.csv
		sleep 1
		./test -r $MATRIX -d $POLDEG  >> test1.csv	
		sleep 1
   done

for((POLDEG=2;POLDEG<=10;POLDEG=POLDEG+1))
  do
     for((MATRIX=1000;MATRIX<=7000;MATRIX=MATRIX+1000))  # me k80 8a pame pio panw 
	   do
	        ./test -r $MATRIX -d $POLDEG  >> test2.csv
			sleep 1
			./test -r $MATRIX -d $POLDEG  >> test2.csv
			sleep 1
			./test -r $MATRIX -d $POLDEG  >> test2.csv
			sleep 1
			./test -r $MATRIX -d $POLDEG  >> test2.csv
			sleep 1
			./test -r $MATRIX -d $POLDEG  >> test2.csv
			sleep 1
						
	   done
  done
   

for((POLDEG=15;POLDEG<=30;POLDEG=POLDEG+5))
  do
     for((MATRIX=1000;MATRIX<=7000;MATRIX=MATRIX+1000))  # me k80 8a pame pio panw 
	   do
	        ./test -r $MATRIX -d $POLDEG  >> test2.csv
			sleep 1
			./test -r $MATRIX -d $POLDEG  >> test2.csv
			sleep 1
			./test -r $MATRIX -d $POLDEG  >> test2.csv
			sleep 1
			./test -r $MATRIX -d $POLDEG  >> test2.csv
			sleep 1
			./test -r $MATRIX -d $POLDEG  >> test2.csv
			sleep 1
						
	   done
  done
  