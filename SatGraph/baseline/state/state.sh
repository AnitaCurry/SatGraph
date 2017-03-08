#!/bin/sh
host=$(hostname)
killall ifstat
killall iostat
rm /home/mapred/share/state/*.if
rm /home/mapred/share/state/*.io
rm /home/mapred/share/share/*.cpu
nohup ifstat -T  -n -z 5 5 > /home/mapred/share/state/$host.if &
nohup iostat -m  -d  5 5 > /home/mapred/share/state/$host.io &
nohup iostat -m  -c 5 5 >  /home/mapred/share/state/$host.cpu &
