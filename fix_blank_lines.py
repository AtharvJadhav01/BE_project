with open('C:/Users/theat/OneDrive/Desktop/BE_Project/my_app/app_work/src/ui/main_window.py', 'rb') as f:
    content = f.read()

# Fix blank lines with spaces - they should be truly empty
# Pattern: \r\n \r\n    def  -> \r\n\r\n    def 
content = content.replace(b'\r\n \r\n    def ', b'\r\n\r\n    def ')

with open('C:/Users/theat/OneDrive/Desktop/BE_Project/my_app/app_work/src/ui/main_window.py', 'wb') as f:
    f.write(content)

print('Fixed!')