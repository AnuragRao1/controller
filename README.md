# Blink Controlled Tetris!

**Two main parts:** 
1. Tetris game (stolen from CS 61B)
2. Blink detection script (used Muse headset with LSL streaming)

You have to run both separately, and then they communicate through a text file (commands.txt) - when a blink is detected from the blinks.py script, it writes either:
- "B" for both eyes 
- "R" for right eye
- "L" for left eye

So the tetris game is always checking that text file for a command:
Right blink = move right, left blink = move left, both = rotate. 
Once it detects a command, it executes it and then deletes it from commands.txt. 

**Things to keep in mind**
- its only really worked for me with intellij? and you need the library from CS 61B. if the tetris part isn't working, I would visit https://fa23.datastructur.es and look through the computer setup. Tetris specifically was lab 12. 
- when you open the project in intellij, go to file->project structure, add an SDK (in project section) and add the library (in libraries)
- muse lsl has only worked on a windows computer, or an intel mac (not m1 or m2)
- the blink detection was pretty good for blinking with both eyes, but it had a lot of trouble differentiating between a left eye blink and a right eye blink. 
- there is a lot of commented-out code in the blinks.py script. I would delete it all, but we were using it to play around with different parameters, such as using different channels, different thresholds... etc, so that might be helpful for you.