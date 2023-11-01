#!/bin/bash

readonly script_name=$(basename $BASH_SOURCE) 

usage() {
cat << USAGE
Usage: bash $script_name [total # of games] [# of parallel simulations]
    [# of games]: how many games to play
    [# of parallel simulations]: how many games to play simultaneously
USAGE
}

game_loop() {
    i=0
    while [ $i -lt $1 ]
    do
        python main.py -g False -s 30 -s100 50 -s10 50 -g_num 0 -p 6
        # printf "$((1 + $RANDOM%7))\n8\n$((1 + $RANDOM%7))" | python main.py -g False -s 30 -s100 50 -s10 50 -g_num 0 
        # printf "$((1 + $RANDOM%7))\n8\n$((1 + $RANDOM%7))" | python clock_game.py -ng True -s $RANDOM
        cat summary_log_nogui.txt >> statistics/output.txt
        ((i++))
    done
}

# if not two arguments are supplied, run usage message       
if [ $# -ne 2 ]; then
	usage
	exit 1
fi

i=$2
games=$1

echo "games="$games"; num_top = 2; -s 30 -s100 50 -s10 50 -g_num 0" > statistics/output.txt

while [ $i -gt 0 ]
do
    num_games=$(($games/$i))
    game_loop $num_games &
    games=$(($games-$num_games))
    ((i--))
done

exit 0