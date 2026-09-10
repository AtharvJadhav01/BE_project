with open('C:/Users/theat/OneDrive/Desktop/BE_Project/my_app/app_work/src/ui/main_window.py', 'r') as f:
    content = f.read()

# Fix the indentation issue
old = '''    def update_state(self, state: str, confidence: float) -> None:
        """Update facial state and confidence.\"\"\"
        self.state_label.setText(f"State: {state}")
        self.confidence_label.setText(f"Confidence: {confidence:.0%}")
        \n
def update_expression(self, expression: str, confidence: float) -> None:
        """Update estimated expression and confidence.\"\"\"
        self.expr_label.setText(f"Estimated Expression: {expression}")
        self.expr_conf_label.setText(f"Expression Confidence: {confidence:.0%}")
        \n
    def update_baseline(self, ready: bool, progress: float = 0.0) -> None:'''

new = '''    def update_state(self, state: str, confidence: float) -> None:
        """Update facial state and confidence.\"\"\"
        self.state_label.setText(f"State: {state}")
        self.confidence_label.setText(f"Confidence: {confidence:.0%}")

    def update_expression(self, expression: str, confidence: float) -> None:
        """Update estimated expression and confidence.\"\"\"
        self.expr_label.setText(f"Estimated Expression: {expression}")
        self.expr_conf_label.setText(f"Expression Confidence: {confidence:.0%}")

    def update_baseline(self, ready: bool, progress: float = 0.0) -> None:'''

if 'def update_state(self, state: str, confidence: float) -> None:' in content:
    content = content.replace(
        '''    def update_state(self, state: str, confidence: float) -> None:
        """Update facial state and confidence.\"\"\"
        self.state_label.setText(f"State: {state}")
        self.confidence_label.setText(f"Confidence: {confidence:.0%}")
        \n
def update_expression(self, expression: str, confidence: float) -> None:
        """Update estimated expression and confidence.\"\"\"
        self.expr_label.setText(f"Estimated Expression: {expression}")
        self.expr_conf_label.setText(f"Expression Confidence: {confidence:.0%}")
        \n
    def update_baseline(self, ready: bool, progress: float = 0.0) -> None:''',
        '''    def update_state(self, state: str, confidence: float) -> None:
        \"\"\"Update facial state and confidence.\"\"\"
        self.state_label.setText(f"State: {state}")
        self.confidence_label.setText(f"Confidence: {confidence:.0%}")

    def update_expression(self, expression: str, confidence: float) -> None:
        \"\"\"Update estimated expression and confidence.\"\"\"
        self.expr_label.setText(f"Estimated Expression: {expression}")
        self.expr_conf_label.setText(f"Expression Confidence: {confidence:.0%}")

    def update_baseline(self, ready: bool, progress: float = 0.0) -> None:'''
    )

with open('C:/Users/theat/OneDrive/Desktop/BE_Project/my_app/app_work/src/ui/main_window.py', 'w') as f:
    f.write(content)
print('Fixed!')