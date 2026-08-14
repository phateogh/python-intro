from quiz_data import get_question
import random, datetime
question_bank = get_question()

#print(question_bank)

random.shuffle(question_bank)
print("=" * 60)
print(" ")
print(question_bank)



question_only = []
answers_only = []
user_respones = []

for question in question_bank:
    #print(question[0])
    question_only.append(question[0])
    answers_only.append(question[1])

    print(question_only)
    print(answers_only)


def ask(question):
    respones =  input(f"{question} :")
    return respones


while True:
    for que in question_bank:
     ans = ask(que)
    user_respones.append(ans)



    break

#correct = 0

#incorrect = 0

for i, corr_ans in enumerate(answers_only):
    for resp in user_respones:
        if resp == corr_ans:
            print("correct")
            #correct += 1
        else:
            print("incorrect")
            #incorrect += 1

        #if i > 5:
            #break