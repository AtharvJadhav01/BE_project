with open('C:/Users/theat/OneDrive/Desktop/BE_Project/my_app/app_work/src/ui/main_window.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Fix line 181 (index 180): blank line between update_state and update_expression - should be empty
lines[180] = '\n'

# Fix line 186 (index 185) - blank line after update_state
lines[185] = '\n'

# Fix line 191 (index 190) - blank line after update_expression
lines[190] = '\n'

with open('C:/Users/theat/OneDrive/Desktop/BE_Project/my_app/app_work/src/ui/main_window.py', 'w', encoding='utf-8') as f:
    f.writelines(lines)

print('Fixed!')