# controller
**Blink Controlled Tetris!**

**Two main parts:** 
1. Tetris game (stolen from CS 61B)
2. Blink detection script

You have to run both, and then they communicate through a separate text file - when a blink is detected from the blinks.py script, it writes either:
- "B" for both eyes 
- "R" for right eye
- "L" for left eye

So the tetris game is always checking that text file for a command. 
Right blink = move right, left blink = move left, both = rotate. 

**Things to keep in mind to run this**
- its only really worked for me with intellij? and you need the library from CS 61B 
- when you open the project in intellij, go to file->project structure, add an SDK (in project section) and add the library (in libraries)
- muse lsl has only worked on a windows computer, or an intel mac (not m1 or m2)