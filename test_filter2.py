import re
def filter_text(raw_text):
    filtered_lines = []
    current_block = []
    
    # We accumulate lines before the first semester
    pre_semester = []
    lines = raw_text.splitlines()
    i = 0
    while i < len(lines) and "Semestre:" not in lines[i]:
        pre_semester.append(lines[i])
        i += 1
    
    filtered_lines.extend(pre_semester)
    
    while i < len(lines):
        if "Semestre:" in lines[i]:
            header = lines[i]
            i += 1
            block = []
            while i < len(lines) and "Semestre:" not in lines[i]:
                # Is it an optativa?
                if re.match(r'^\S{6} ', lines[i]):
                    i += 1
                    continue
                block.append(lines[i])
                i += 1
            
            # Check if block has courses
            has_courses = any(re.match(r'^ {7,}\w+', l) for l in block)
            if has_courses:
                filtered_lines.append(header)
                filtered_lines.extend(block)
        else:
            filtered_lines.append(lines[i])
            i += 1
            
    return "\n".join(filtered_lines)

with open('materias/LCC 440 3.txt', 'r') as f:
    text = f.read()

res = filter_text(text)
print("Semestre 11 in text?", "Semestre:11" in text)
print("Semestre 11 in filtered?", "Semestre:11" in res)
print("08PI08 in filtered?", "08PI08" in res)
