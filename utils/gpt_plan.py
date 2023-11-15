import os
import ast
import json
import openai
import argparse
import time
import jsonlines
from datetime import datetime, timedelta

os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'

# Load your API key from an environment variable or secret management service
openai.api_key = "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"

model_list = openai.Model.list()
for model in model_list['data']:
    if model['id'][:3] == 'gpt':
        print(model['id'])

system_prompt = "Assuming you have a specific personality, please plan your daily activities, especially your interactions with furniture in your home, and return it to me in JSON format.Your areas of daily activities are limited to this room list: [livingroom, bedroom, bathroom, kitchen, study, hall].The interaction furniture of your daily activities are limited to a furniture dictionary, which the key is the furniture name, the value is the furniture state class. All of your interaction furniture should be chosen from this furniture dictionary:{    air conditioner in bedroom: [electrical device],    air conditioner in livingroom: [electrical device],    apple: [food, tableware],    apricot: [food, tableware],    BarStool 01: [big furniture],    BarStool 02: [big furniture],    BarStool 03: [big furniture],    Bathtub: [usable device],    beef: [food, tableware],    bowl 01: [tableware],    bowl 02: [tableware],    bowl 03: [tableware],    bowl 04: [tableware],    bowl 05: [tableware],    bowl 06: [tableware],    bowl 07: [tableware],    bowl 08: [tableware],    bowl 09: [tableware],    bowl 10: [tableware],    bowl 11: [tableware],    bowl 12: [tableware],    bowl 13: [tableware],    bowl 14: [tableware],    bowl 15: [tableware],    bowl 16: [tableware],    bread: [food, tableware],    carrot: [food, tableware],    cell phone: [portable device,cell phone],    chair in bathroom: [big furniture],    chair in hall: [big furniture],    chair in kitchen 01: [big furniture],    chair in kitchen 02: [big furniture],    chair in kitchen 03: [big furniture],    chair in kitchen 04: [big furniture],    chair in kitchen 05: [big furniture],    chair in kitchen 06: [big furniture],    chair in livingroom beside guitar: [big furniture],    chair in livingroom beside window: [big furniture],    chair in study: [big furniture],    chopsticks 01: [portable tableware],    chopsticks 02: [portable tableware],    cleasing milk: [portable device],    Closet 01: [door],    Closet 02: [door],    Closet 03: [door],    cloth 01: [portable device],    cloth 02: [portable device],    cloth 03: [portable device],    coffe powder: [food],    coffee machine: [electrical device],    cooked meal: [meal,tableware],    cooking appliance 01: [cook appliance],    cooking appliance 02: [cook appliance],    cooking appliance 03: [cook appliance],    cooking appliance 04: [cook appliance],    cooking appliance 05: [cook appliance],    cooking appliance 06: [cook appliance],    cooking knife: [cook appliance],    cup 01: [portable tableware],    cup 02: [portable tableware],    cup 03: [portable tableware],    cup 04: [portable tableware],    cup 05: [portable tableware],    cup 06: [portable tableware],    cup 07: [portable tableware],    cup 08: [portable tableware],    cup 09: [portable tableware],    cup 10: [portable tableware],    cup 11: [portable tableware],    cup 12: [portable tableware],    cup 13: [portable tableware],    cup 14: [portable tableware],    cup 15: [portable tableware],    cup 16: [portable tableware],    cup 17: [portable tableware],    cup 18: [portable tableware],    cup 19: [portable tableware],    cup 20: [portable tableware],    cup 21: [portable tableware],    curtain bedroom: [door],    cutting board: [usable device],    disc player: [electrical device],    donut: [food, tableware],    door bathroom: [door],    door bedroom: [door],    drawer kitchen 01: [door],    drawer kitchen 02: [door],    drawer kitchen 03: [door],    drawer kitchen 04: [door],    drawer kitchen 05: [door],    drawer kitchen 06: [door],    drawer kitchen 07: [door],    drawer kitchen 08: [door],    drawer kitchen 09: [door],    drawer kitchen 10: [door],    drawer kitchen 11: [door],    drawer TV left: [door],    drawer TV right: [door],    egg: [food,tableware],    extractor hood: [electrical device],    fork 01: [portable tableware],    fork 02: [portable tableware],    fridge: [door],    game controller: [portable device],    glasses: [portable device],    guitar: [portable device],    hard disk drive: [portable device],    headphone: [portable device],    Induction Cooker: [electrical device],    jug: [container],    kettle for boil water: [electrical device],    kettle for contain water: [container],    knife 01: [portable tableware],    knife 02: [portable tableware],    lamb: [food,tableware],    lamp 01: [electrical device],    lamp 02: [electrical device],    lamp in bedroom 01: [electrical device],    lamp in bedroom 02: [electrical device],    lamp in bedroom 03: [electrical device],    lamp in hall 01: [electrical device],    lamp in hall 02: [electrical device],    lamp in livingroom: [electrical device],    landline in livingroom: [usable device],    landline in bedroom: [usable device],    laptop: [electrical device, big furniture],    lemon: [food, tableware],    light in kitchen: [electrical device],    light in livingroom: [electrical device],    light in study: [electrical device],    light in bedroom: [electrical device],    main door: [door],    microwave: [electrical device],    milk: [food,tableware],    mop: [portable device],    Nintendo switch: [portable device],    oil: [food],    oven: [electrical device],    pan 01: [usable device],    pan 02: [usable device],    PC: [electrical device],    piano: [usable device],    plate 01: [tableware],    plate 02: [tableware],    plate 03: [tableware],    plate 04: [tableware],    plate 05: [tableware],    plate 06: [tableware],    plate 07: [tableware],    plate 08: [tableware],    plate 09: [tableware],    plate 10: [tableware],    plate 11: [tableware],    plate 12: [tableware],    plate 13: [tableware],    plate 14: [tableware],    plate 15: [tableware],    plate 16: [tableware],    plate 17: [tableware],    plate 18: [tableware],    plate 19: [tableware],    plate 20: [tableware],    plate 21: [tableware],    plate 22: [tableware],    plate 23: [tableware],    plate 24: [tableware],    plate 25: [tableware],    plate 26: [tableware],    plate for fruit 01: [tableware],    plate for fruit 02: [tableware],    pork: [food,tableware],    pot: [usable device],    potato: [food,tableware],    powder: [food],    Printer: [electrical device],    rag 01: [portable device],    rag 02: [portable device],    rag 03: [portable device],    rice: [food],    salt: [food],    shoes 01: [portable device],    shoes 02: [portable device],    shoes 03: [portable device],    shower: [electrical device],    smart speaker: [electrical device],    soy sauce: [food],    spatula 01: [cook appliance],    spatula 02: [cook appliance],    spatula 03: [cook appliance],    spoon 01: [portable tableware],    spoon 02: [portable tableware],    sugar: [food],    tea: [food],    tea pot: [usable device],    toilet: [usable device],    tomato: [food, tableware],    toothbrush: [portable device],    toothbrush cup: [portable device],    toothpaste: [usable device],    towel 01: [portable device],    towel 02: [portable device],    TV: [electrical device],    vegetable: [food,tableware],    vinegar: [food],}Your actions are limited to an action dictionary, which the key is the furniture state class, the value is the possible action. All actions should be chosen from the value list of the corresponding key:{    electrical device: [turn on,turn off],    cell phone: [charge, uncharge],    big furniture: [move],    food: [eat,consume,cook,buy],    meal: [store up,take out,eat,buy],    tableware: [move],    portable tableware: [move,pick up,put down],    portable device: [pick up,put down],    door: [open, close],    cook appliance: [move,pick up,put down],    usable device: [use,stop using],    container: [full, empty],    personal state: [wake up,sleep,leave home,come home,sit,stand up,go to *(room)],}Furniture states are limited to this state dictionary, which the key is the furniture state class, the value is the possible states:{    electrical device: [on, off],    cell phone: [charging, not charging],    big furniture: [livingroom, bedroom, bathroom, kitchen, study, hall],    food: {in fridge:10, on kitchen shelf:0, in sink:0, on kitchen counter:0, on dining table:0, in microwave:0, in oven:0},    meal:  {in fridge:10, on kitchen shelf:0, in sink:0, on kitchen counter:0, on dining table:0, in microwave:0, in oven:0},    tableware: [in fridge, on kitchen shelf, in sink, on kitchen counter, on dining table, in microwave, in oven],    portable tableware: [with human,in fridge, on kitchen shelf, in sink, on kitchen counter, on dining table, in microwave, in oven],    portable device: [with human, placed],    door: [open, close],    cook appliance: [with human, on kitchen shelf, in sink, on kitchen counter, on dining table],    usable device: [using, free],    container: [full, empty],}Your daily activities plan must follow the predefined lifestyle and habits based on your specific personality and hobbies, sometimes basic information also needs to be considered.In the JSON, outline targets and their corresponding actions with duration time (in minutes, no less than 1 minute) sequentially. Each target can have one or more associated actions. The start_time of an action in JSON should be the previous action's start_time plus duration.Forecast changes in the object's state caused by each action. Only objects with altered states will be included in the JSON.When you eat food or meal, their number should reduce at least 1. You cannot eat them if the number is 0.Each food has 10 in fridge. When food number reach 0, you should buy them before eating. When you buy some food, the number of them increased to 10.meal is 0 initially. Cook and buy action will increase its number by 1. if the number reduced to 0, you should cook or buy before eat."

def ft_data_format(data):
    response_content = data['messages'][2]['content']
    # response_content = response_content.replace('\\', '').replace('\n', '').replace('\"', '"').replace('，', ',')
    response_content = ast.literal_eval(response_content)
    return response_content


def test(args, day_flag="d001-002", p_str="p01", start_day="2023.07.11"):
    prompt_base = open("prompt_gpt4.txt", "r").read()
    personality = open(f"{p_str}.txt", "r").read()
    prompt = f"{prompt_base}{personality}\n\nPlease plan your activities for the next {args.e} days, start from {start_day}:\n["  # default generate 2 days at a time
    # prompt = f"\n{personality}\n\nPlease plan your activities for the next {args.e} days, start from {start_day}:\n["  # default generate 2 days at a time
    print(prompt)

    try:
        response = openai.ChatCompletion.create(model="gpt-4-0314",  # model="gpt-3.5-turbo-16k",
                                                messages=[{
                                                    "role": "user",
                                                    "content": prompt
                                                }],
                                                max_tokens=3402, #10240,
                                                temperature=args.t,
                                                n=1,
                                                frequency_penalty=0,
                                                presence_penalty=0.05)

        print(response, '\n')
        response_content = '[' + response['choices'][0]['message']['content']
        response_content = response_content.replace('\\', '').replace('\n', '').replace('\"', '"').replace('，', ',')
        response_content = ast.literal_eval(response_content)
        with open(f"{p_str}_{day_flag}_{args.t}_{args.f}.json", "w") as file:
            json.dump(response_content, file, indent=4)
        return True
    except Exception as e:
        print(str(e))
        return False

         
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--p', type=str, default = 'p01')
    parser.add_argument('--t', type=float, default=1.115)
    # parser.add_argument('--t', type=float, default=1.2)
    parser.add_argument('--f', type=str, default = '')  # comment or flag
    parser.add_argument('--s', type=int, default = 1)  # start_day
    parser.add_argument('--d', type=int, default = 2)  # end_day
    parser.add_argument('--e', type=int, default = 2)  # generate how many days at a time
    args = parser.parse_args()

    start_date = datetime.strptime("2023.07.11", "%Y.%m.%d")


    if "-" in args.p:
        start_p, end_p = args.p.split("-")
        start_p = start_p.replace("p", "")
        participant_list = range(int(start_p), int(end_p) + 1)
    else:
        start_p = int(args.p.replace("p", ""))
        participant_list = range(start_p, start_p + 1)
    
    for participant in participant_list:
        p_str = "p{0}".format(str(participant).zfill(2))
        print(f"### Start generating {p_str}...\n")
    
        day_num = args.s - 1
        while day_num < args.d:
            if args.e == 1:
                day_flag = "d{0}".format(str(day_num+1).zfill(3))
            else:
                day_flag = "d{0}-{1}".format(str(day_num+1).zfill(3), str(day_num+args.e).zfill(3))
            
            # calculate start_day
            cur_start_date = start_date + timedelta(days=day_num)
            start_day = cur_start_date.strftime("%Y.%m.%d")
            
            # generate data
            generate_flag = test(args, day_flag, p_str, start_day)
            
            if generate_flag:
                # successfully generated. go to next day
                day_num += args.e
            else:
                # generate failed. try again.
                print("Generation of %s failed. Regenerating...\n" % day_flag)
            time.sleep(100)

