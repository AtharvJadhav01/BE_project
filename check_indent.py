with open('C:/Users/theat/OneDrive/Desktop/BE_Project/my_app/app_work/src/ui/main_window.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

for i in range(178, 195):
    if i < len(lines):
        line = lines[i]
        indent = len(line) - len(line.lstrip())
        first_char = repr(line[0]) if line else 'empty'
        print(f'{i+1:4d}: indent={indent} first={repr(line[0]) if line else "empty"} repr={repr(line[:60])}')