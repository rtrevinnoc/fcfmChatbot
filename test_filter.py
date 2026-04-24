import re
def filter_text(raw_text):
    filtered_lines = []
    current_block = []
    in_semester = False
    for line in raw_text.splitlines():
        if "Semestre:" in line:
            if current_block:
                has_courses = any(l.strip() != "" for l in current_block[1:])
                if has_courses or not in_semester:
                    filtered_lines.extend(current_block)
            current_block = [line]
            in_semester = True
        else:
            is_optativa = re.match(r'^\S{6} ', line)
            if not is_optativa:
                current_block.append(line)
    if current_block:
        has_courses = any(l.strip() != "" for l in current_block[1:])
        if has_courses or not in_semester:
            filtered_lines.extend(current_block)
    return "\n".join(filtered_lines)

with open('materias/LCC 440 3.txt', 'r') as f:
    text = f.read()

res = filter_text(text)
print("Semestre 11 in text?", "Semestre:11" in text)
print("Semestre 11 in filtered?", "Semestre:11" in res)
print("08PI08 in filtered?", "08PI08" in res)
