'''
Test Cases for v1 will go here
'''
from benesse_rule_based_code import *
import stanfordnlp
import json

def eq(input, target, comparison):
    no_failed_cases = comparison(input, target)
    return no_failed_cases

def json_comparison(input_json, expected_json):
	count = 0
	for i in range(len(input_json)):
		try:
			assert (input_json[i] == expected_json[i])
		except:
			count += 1
			continue
	return count


def read_file(path):
	file = open(path, "rt")
	text = []
	for l in file.readlines():
		text.append(l)
	return text

def test_capitalize(filepath,nlp,d):
    # read filepath
    text = read_file(filepath)
    #write_file = open("test_capitalize_samples_expected.json","a")
    sample_count = 0
    total_errors = []
    for t in text:
    	sample_count += 1
    	#print("Sample Number",sample_count)
    	doc = nlp(t)
    	idx = 0
    	errors = {'Sample No':sample_count,'Detections': []}
    	for sentence in doc.sentences:
            res = CapitalizeError(idx, sentence)
            for r in res:
                #errors['Detections'].append({'Sentence ID':r[0],'Sentence Text':sentence.text,'Word Index':r[1].index ,'Word String':r[1].text,'Error Type':r[2]})
                t = ""
                for w in sentence.words:
                    t += w.text + " "
                d.append([sample_count,r[0],t,r[1].index ,r[1].text,r[2]])
            idx += 1
    	json_str = json.dumps(errors)
    	print(json_str)
    	total_errors.append(errors)
    #json.dump(total_errors,write_file,indent = 2)
    input_json = total_errors
    with open('test_capitalize_samples_expected.json') as f:
    	expected_json = json.load(f)

    count = eq(input_json,expected_json,json_comparison)
    print("No of test failures for Capitalisation Error: ", count)
    return d

def test_pluralize(filepath,nlp,d):
	# read filepath
    text = read_file(filepath)
    #write_file = open("test_pluralize_samples_expected.json","a")
    sample_count = 0
    total_errors = []
    for t in text:
    	sample_count += 1
    	#print("Sample Number",sample_count)
    	doc = nlp(t)
    	idx = 0
    	errors = {'Sample No':sample_count,'Detections': []}
    	for sentence in doc.sentences:
            res = pluralizationError(idx, sentence)
            for r in res:
                t = ""
                for w in sentence.words:
                    t += w.text + " "
                #errors['Detections'].append({'Sentence ID':r[0],'Sentence Text':sentence.text,'Word Index':r[1].index ,'Word String':r[1].text,'Error Type':r[2]})
                d.append([sample_count,r[0],t,r[1].index ,r[1].text,r[2]])
            idx += 1
    	json_str = json.dumps(errors)
    	print(json_str)
    	total_errors.append(errors)
    #json.dump(total_errors,write_file,indent = 2)
    input_json = total_errors
    with open('test_pluralize_samples_expected.json') as f:
    	expected_json = json.load(f)

    count = eq(input_json,expected_json,json_comparison)
    print("No of test failures for Pluralization Error: ", count)
    return d

def test_because(filepath,nlp,d):
	# read filepath
    text = read_file(filepath)
    #write_file = open("test_because_samples_expected.json","a")
    sample_count = 0
    total_errors = []
    for t in text:
    	sample_count += 1
    	#print("Sample Number",sample_count)
    	doc = nlp(t)
    	idx = 0
    	errors = {'Sample No':sample_count,'Detections': []}
    	for sentence in doc.sentences:
            res = becauseError(idx, sentence)
            for r in res:
                #errors['Detections'].append({'Sentence ID':r[0],'Sentence Text':sentence.text,'Word Index':r[1].index ,'Word String':r[1].text,'Error Type':r[2]})
                t = ""
                for w in sentence.words:
                    t += w.text + " "
                d.append([sample_count,r[0],t,r[1].index ,r[1].text,r[2]])
            idx += 1
    	json_str = json.dumps(errors)
    	print(json_str)
    	total_errors.append(errors)
    #json.dump(total_errors,write_file,indent = 2)
    input_json = total_errors
    with open('test_because_samples_expected.json') as f:
    	expected_json = json.load(f)

    count = eq(input_json,expected_json,json_comparison)
    print("No of test failures for Because Error: ", count)
    return d

d= []

nlp = stanfordnlp.Pipeline(processors = "tokenize,mwt,lemma,pos",use_gpu=True)

d = test_capitalize("test_capitalize_samples.txt",nlp,d)
d = test_pluralize("test_pluralize_samples.txt",nlp,d)
d = test_because("test_because_samples.txt",nlp,d)

import pandas as pd

df = pd.DataFrame(d, columns = ['Sample ID', 'Sentence ID','Sentence Text','Word Index','Word Text','Error Type'])
df.to_csv("error_detections.csv")



