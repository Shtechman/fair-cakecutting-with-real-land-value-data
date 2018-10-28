# PARAMS:
## filename - relative path to the file with the data to plot.
## xcolumn - 1 is number of agents, 2 is log num of agents, 3 is amplitude of noise in utilities.
## xlabel - label for the x axis.

set autoscale x
set autoscale y
# set yrange [-1:3]
set xlabel xlabel
set ylabel 'gain/loss'
set key left top

set terminal pngcairo
set output sprintf('%s.png',filename)

f1(x)=a1*x+b1
f2(x)=a2*x+b2
f3(x)=a3*x+b3
f4(x)=a4*x+b4
f5(x)=a5*x+b5
f6(x)=a6*x+b6
zero(x)=0
fit f1(x) filename using xcolumn:4 via a1,b1
fit f2(x) filename using xcolumn:5 via a2,b2
fit f3(x) filename using xcolumn:6 via a3,b3
fit f4(x) filename using xcolumn:7 via a4,b4
fit f5(x) filename using xcolumn:8 via a5,b5
fit f6(x) filename using xcolumn:9 via a6,b6

plot sin(x)
