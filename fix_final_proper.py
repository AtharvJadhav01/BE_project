with open('C:/Users/theat/OneDrive/Desktop/BE_Project/my_app/app_work/src/ui/main_window.py', 'rb') as f:
    content = f.read()

# Fix the blank lines - they have 1 space instead of being truly empty
# Find and fix the three problematic blank lines

# Pattern 1: blank line after update_state (line 186)
old1 = b'\r\n \r\n    def update_expression'
new1 = b'\r\n\r\n    def update_expression'

# Pattern 2: blank line after update_expression (line 191)
old2 = b'\r\n \r\n    def update_baseline'
new2 = b'\r\n\r\n    def update_baseline'

# Also fix the blank line after update_state (line 186)
old3 = b'\r\n \r\n    def update_expression'
new3 = b'\r\n\r\n    def update_expression'

# Actually let's be more precise - find the exact patterns
with open('C:/Users/theat/OneDrive/Desktop/BE_Project/my_app/app_work/src/ui/main_window.py', 'rb') as f:
    content = f.read()

# Fix: blank lines should have no spaces
# Pattern: \r\n \r\n    def -> \r\n\r\n    def
content = content.replace(b'\r\n \r\n    def ', b'\r\n\r\n    def ')

with open('C:/Users/theat/OneDrive/Desktop/BE_Project/my_app/app_work/src/ui/main_window.py', 'wb') as f:
    f.write(content)

print('Fixed!')