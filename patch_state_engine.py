import re

with open("state_engine.py", "r") as f:
    content = f.read()

# Remove override setter
content = re.sub(r'    @active_positions\.setter.*?        with self\.lock:\n            self\._active_positions_override = value\n', '', content, flags=re.DOTALL)

with open("state_engine.py", "w") as f:
    f.write(content)
