
import numpy as np


raw_students = [
    "zg0505_짱구",
    "cs0822_철수",
    "mg0910_맹구",
    "yr0605_유리",
    "hi0205_훈이"
]

raw_scores = np.array([
    [95, 82, 88],
    [70, 65, 90],
    [98, 100, 92],
    [60, 75, 80],
    [81, 88, 84]
])

subjects = ["국어", "수학", "과학"]



student_names = [
    student_record.split("_")[1]
    for student_record in raw_students
]

print("1. 학생 명단:", ", ".join(student_names))



all_scores = raw_scores.flatten()

print("2. 전산 시스템에 입력된 모든 성적:")
print(all_scores)



scores_over_85 = np.sum(all_scores >= 85)

print(f"3. 전체 15개 성적 중 85점 이상 개수: {scores_over_85}개")



student_average_scores = np.mean(raw_scores, axis=1)

print("4. 학생별 평균 점수:")
for student_name, average_score in zip(student_names, student_average_scores):
    print(f"{student_name} {average_score:.2f}점", end="  ")
print()



top_student_index = np.argmax(student_average_scores)
top_student_name = student_names[top_student_index]
top_student_average = student_average_scores[top_student_index]

print(f"5. 최고 평균 수석 학생: {top_student_name} ({top_student_average:.2f}점)")



overall_max_score = np.max(raw_scores)
overall_max_index = np.argmax(raw_scores)

student_index = overall_max_index // len(subjects)
subject_index = overall_max_index % len(subjects)

overall_max_student = student_names[student_index]
overall_max_subject = subjects[subject_index]

print(f"6. 단일 과목 최고 점수: {overall_max_score}점 "
      f"(과목: {overall_max_subject}, 학생: {overall_max_student})")
