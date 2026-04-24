import re
import os

MATERIAS_DIR = "materias"
CAREER_FILE_MAP = {
    "computacionales": ["LCC 440 3.txt", "LCC 420 3.txt"],
}

def get_career_materias_context(query: str) -> str:
    q = query.lower()
    wants_optativas = "optativa" in q

    for keyword, filenames in CAREER_FILE_MAP.items():
        if keyword in q:
            for fname in filenames:
                fpath = os.path.join(MATERIAS_DIR, fname)
                if os.path.exists(fpath):
                    with open(fpath, "r", encoding="utf-8") as f:
                        raw_text = f.read()
                        
                        if wants_optativas:
                            return raw_text

                        filtered_lines = []
                        lines = raw_text.splitlines()
                        i = 0
                        
                        optativa_pattern = re.compile(r'^\S{6,10} ')
                        course_pattern = re.compile(r'^ {7,}\w+')

                        while i < len(lines) and "Semestre:" not in lines[i]:
                            filtered_lines.append(lines[i])
                            i += 1
                        
                        while i < len(lines):
                            if "Semestre:" in lines[i]:
                                header = lines[i]
                                i += 1
                                block = []
                                while i < len(lines) and "Semestre:" not in lines[i]:
                                    if optativa_pattern.match(lines[i]):
                                        i += 1
                                        continue
                                    block.append(lines[i])
                                    i += 1
                                
                                has_courses = any(course_pattern.match(l) for l in block)
                                if has_courses:
                                    filtered_lines.append(header)
                                    filtered_lines.extend(block)
                            else:
                                filtered_lines.append(lines[i])
                                i += 1
                                
                        return "\n".join(filtered_lines)
    return ""

query = "qué materias se ofrecen en Ciencias Computacionales"
res = get_career_materias_context(query)
print("Semestre 11 in filtered?", "Semestre:11" in res)
print("Fundamentos de administración in filtered?", "Fundamentos de administración" in res)
print("Semestre 06 in filtered?", "Semestre:06" in res)
