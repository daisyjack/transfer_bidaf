#-*- coding:utf-8 -*-
#!/usr/bin/python
import time

class Levenshtein():

    def __init__(self):
        pass

    def levenshtein(self,first,second):
        if len(first) > len(second):
            first,second = second,first
        if len(first) == 0:
            return len(second)
        if len(second) == 0:
            return len(first)
        first_length = len(first) + 1
        second_length = len(second) + 1
        distance_matrix = [list(range(second_length)) for x in range(first_length)]
        #print distance_matrix
        for i in range(1,first_length):
            for j in range(1,second_length):
                deletion = distance_matrix[i-1][j] + 1
                insertion = distance_matrix[i][j-1] + 1
                substitution = distance_matrix[i-1][j-1]
                if first[i-1] != second[j-1]:
                    substitution += 1
                distance_matrix[i][j] = min(insertion,deletion,substitution)
        return distance_matrix[first_length-1][second_length-1]

class patternLearning(object):
    def __init__(self):
        self.argument_tag = ["ASR","ASE","MEY"]
        # self.argument_tag = ["FIRST","SECOND"]

        self.ner_tag = ["com", "loc", "per", "product","org","time","m","prot"]
        self.punctuation_tag = ["x"]
        self.levenshtein = Levenshtein()
        self.ptn_edit_distinct_limit = 20

    def soft_match_score(self, segment1, segment2):
        if len(segment1) == 1 and len(segment2)==1:
            if segment1[0] == segment2[0]:
                return [0, segment1[0]]
            else:
                return [10, None]
        elif len(segment1)==1 or len(segment2)==1:
            return [10, None]

        word1, tag1 = segment1
        word2, tag2 = segment2

        # token equal
        if word1 == word2:
            return [0, None]

        # special token
        if word1 == word2 and tag1 in self.punctuation_tag:
            return [5, None]

        # tag equal
        if tag1 == tag2:
            if tag1 in self.ner_tag:
                return [5, tag1]
            else:
                return [10, None]
        # not match
        return [20, None]


    def merge_lcs_result(self, sentence_pattern):
        result = []
        prev_is_any = False
        for item in sentence_pattern:
            if len(item) == 1:
                result.append(item)
                prev_is_any = False
            elif len(item) == 2:
                result.append(item)
                prev_is_any = False
            elif len(item) == 3:
                if prev_is_any:
                    result[-1][1].append([item[1], item[2]])
                else:
                    result.append([item[0],[[item[1], item[2]]]])
                prev_is_any = True

        # for i in result:
        #     if len(i) == 1:
        #         print i[0],
        #     elif len(i) == 2:
        #         if type(i[1]) == list:
        #             rr = [j[0]+"_"+j[1] for j in i[1]]
        #
        #             print i[0]+"#"+",".join(rr),
        #         else:
        #             print i[0]+"_"+i[1],
        return result



    def soft_match_lcs(self, array1, size1, array2, size2):
        synonym_dict_array1 = {}
        match_scores = [[]for i in range(size1)]
        for i in range(size1):
            match_scores[i] = [0 for j in range(size2)]
        for i in range(size1):
            a=[]
            for j in range(size2):
                score, synonym_index = self.soft_match_score(array1[i], array2[j])

                if synonym_index!=None:
                    synonym_dict_array1[i] = synonym_index
                match_scores[i][j] = 10 - score
                a.append(str(match_scores[i][j]))

        # soft lcs
        common = [[] for i in range(size1 + 1)]
        for i in range(size1+1):
            common[i] = [0 for j in range(size2 + 1)]
        for i in range(size1):
            for j in range(size2):
                if match_scores[i][j]>0:
                    common[i+1][j+1] = common[i][j] + match_scores[i][j]
                else:
                    common[i+1][j+1] = max(common[i+1][j], common[i][j+1])

        max_score = common[size1][size2]

        iidx1 = size1
        iidx2 = size2
        result = []
        flag = True
        while iidx1 >=0 and iidx2 >=0:
            if common[iidx1][iidx2] == 0:
                break
            if match_scores[iidx1-1][iidx2-1] > 0:
                synonym_str = ""
                if iidx1-1 in synonym_dict_array1:
                    synonym_str = synonym_dict_array1[iidx1-1]
                if synonym_str in self.ner_tag:
                    result.append([synonym_str])
                else:
                    result.append(array1[iidx1-1])
                iidx1 -= 1
                iidx2 -= 1
            else:
                if common[iidx1-1][iidx2] > common[iidx1][iidx2-1]:
                    iidx1 -= 1

                    if len(array1[iidx1])==1:
                        flag = False
                        break
                    result.append(["*",array1[iidx1][0],array1[iidx1][1]])
                else:
                    iidx2 -= 1
        if not flag:
            return None
        result.reverse()

        return self.merge_lcs_result(result)

        # for i in result:
        #     if len(i)==1:
        #         print i,
        #     elif len(i)==2:
        #         print i[0]+"_"+i[1],
        #     elif len(i)==3:
        #         print i[0]+"_"+i[1]+"_"+i[2],
        # print "\n",max_score

        # return result, max_score

    def is_valid_pattern(self, ptn):
        argments = set()
        for item in ptn:
            if len(item) == 1:
                if item[0] in self.argument_tag:
                    argments.add(item[0])
        if len(argments) != len(self.argument_tag):
            return False
        return True

    def sentence_to_segment(self, sen):
        a = []
        for i in sen.split(" "):
            if "_" not in i:
                a.append([i])
            else:
                c,b = i.split("_")
                if b in self.argument_tag or b in self.ner_tag:
                    a.append([b])
                else:
                    a.append([c,b])
        # for j in a:
        #     if len(j) == 1:
        #         print j[0],
        #     elif len(j) == 2:
        #         print j[0]+"_"+j[1],
        return a

    def satisfy_candidate_edit_distance(self, alist, blist):
        aword = [i[0] for i in alist]
        bword = [i[0] for i in blist]
        dis = self.levenshtein.levenshtein(aword, bword)
        if dis <= self.ptn_edit_distinct_limit:
            return True
        else:
            return False

    def sentence_to_pattern(self, sen1, sen2):
        alist = self.sentence_to_segment(sen1)
        blist = self.sentence_to_segment(sen2)
        satisfy_edit_dis = self.satisfy_candidate_edit_distance(alist, blist)
        if not satisfy_edit_dis:
            return None
        res = self.soft_match_lcs(alist, len(alist), blist, len(blist))
        if not res:
            return None
        if self.is_valid_pattern(res):
            return res
        return None


    def get_pattern_core(self, pattern):
        res = []
        for i in pattern:
            if len(i) == 1:
                res.append(i[0])
            elif len(i) == 2:
                res.append(i[0])

        # for i in pattern:
        #     if len(i) == 1:
        #         print i[0],
        #     elif len(i) == 2:
        #         if type(i[1]) == list:
        #             rr = [j[0]+"_"+j[1] for j in i[1]]
        #
        #             print i[0]+"#"+",".join(rr),
        #         else:
        #             print i[0]+"_"+i[1],
        # print "\n"
        return ",".join(res)


    def pattern_mergeing(self, ptn1, ptn1_fre, ptn2, ptn2_fre):
        #频次相同的模板集合中，如果能匹配模板a的一定能匹配模板b，则保留模板a
        new_ptn = []
        not_match = False

        ptn_index1 = 0
        ptn_index2 = 0
        while ptn_index1 < len(ptn1) and ptn_index2 < len(ptn2):
            if not_match:
                break
            if ptn1[ptn_index1] == ptn2[ptn_index2]:
                new_ptn.append(ptn1[ptn_index1])
                ptn_index1 += 1
                ptn_index2 += 1
                continue
            else:
                if ptn_index1+1<len(ptn1) and ptn_index2+1<len(ptn2):
                    if ptn1[ptn_index1+1] == ptn2[ptn_index2+1]:
                        if ptn1[ptn_index1] !="*" and ptn2[ptn_index2]!="*":
                            new_ptn.append("<%s|%s>" %(ptn1[ptn_index1], ptn2[ptn_index2]))
                        elif ptn1[ptn_index1] =="*":
                            new_ptn.append("(%s|*)" %(ptn2[ptn_index2]))
                        elif ptn2[ptn_index2]=="*":
                            new_ptn.append("(%s|*)" %(ptn1[ptn_index1]))
                        ptn_index1+=1
                        ptn_index2+=1
                    else:
                        if ptn1[ptn_index1+1] == ptn2[ptn_index2] and ptn1[ptn_index1] == ptn2[ptn_index2+1]:
                            if ptn1[ptn_index1] == "*":
                                new_ptn.append("(*),(%s),(*)" %(ptn1[ptn_index1+1]))
                                ptn_index1 += 2
                                ptn_index2 += 2

                            elif ptn1[ptn_index1+1] =="*":
                                new_ptn.append("(*),(%s),(*)" %(ptn1[ptn_index1]))
                                ptn_index1 += 2
                                ptn_index2 += 2
                            else:
                                not_match = True
                                break
                        elif ptn1[ptn_index1+1] == ptn2[ptn_index2]:
                            new_ptn.append("(%s)"%(ptn1[ptn_index1]))
                            ptn_index1 += 1
                        elif ptn1[ptn_index1] == ptn2[ptn_index2+1]:
                            new_ptn.append("(%s)" %(ptn2[ptn_index2]))
                            ptn_index2 += 1
                        else:
                            not_match = True
                            break
                else:
                    if ptn_index1+1<len(ptn1):
                        new_ptn.append("(%s)" %(ptn1[ptn_index1+1]))
                        break
                    elif ptn_index2+1<len(ptn2):
                        new_ptn.append("(%s)" %(ptn2[ptn_index2+1]))
                        break
                    else:
                        ptn_index1+=1
                        ptn_index2+=1

        if not not_match:
            return new_ptn
        return None





ptn_learn = patternLearning()

a = "中航沈飞股份有限公司_com 控股_v 子公司_n 柳州乘龙专用车有限公司_ASR 与_p 核心_n 经销商_n 广西东葵贸易有限公司_ASE 合作_vn 开展_v 敞口_v 银行_n 承兑_v " \
  "业务_n ，_x 总_b 授信额度_n 4500万元_MEY ，_x 由_p 柳州乘龙专用车有限公司_ASR 提供_v 担保_v 。_x"
b = "中航沈飞股份有限公司_com 控股_v 子公司_n 柳州乘龙专用车有限公司_ASR 与_p 核心_n 经销商_n 柳州市广捷汽车贸易有限公司_ASE 合作_vn 开展_v 敞口_v 银行_n " \
    "承兑_v 业务_n ，_x 总_b 授信额度_n 1000万元_MEY ，_x 由_p 柳州乘龙专用车有限公司_ASR 提供_v 担保_v 。_x"
c = "中航沈飞股份有限公司_com 控_v 来自_v 柳州乘龙专用车有限公司_ASR 经销商_n 柳州市广捷汽车贸易有限公司_ASE 合作_vn 开展_v 敞口_v 银行_n " \
    "承兑_v 业务_n ，_x 总_b 授信额度_n 1000万元_MEY ，_x 由_p 柳州乘龙专用车有限公司_ASR 提供_v 担保_v 。_x"


res = ptn_learn.sentence_to_pattern(a, c)
print(res)

'''
pattern_dict = {}
pattern_count = {}

sentences = []
for line in open("./resource/assurance_train_segment"):
    line = line.strip()
    flag = True
    for i in line.split(" "):
        if "_" not in i:
            flag = False
    if flag:
        sentences.append(line)
count = 0
flag = True

pattern_full_count = []
for idxi in range(len(sentences)):
    sentencei = sentences[idxi]
    count += 1
    print count
    # sentencei = a
    if not flag:
        break
    pattern_list = set()
    for idxj in range(len(sentences)):
        if idxi == idxj:
            continue
        sentencej = sentences[idxj]

        # sentencej = b
        # print sentencei
        # print sentencej
        # count -= 1
        # if count %100==0:
        #     print count
        # if count < 0:
        #     flag = False
        res = ptn_learn.sentence_to_pattern(sentencei, sentencej)
        # print res
        if res and len(res)>5:
            ptn_core = ptn_learn.get_pattern_core(res)
            pattern_list.add(ptn_core)
            pattern_full_count.append(res)
            # print sentencei
            # print sentencej
            # print ptn_core
            # print "======"
            # pattern_count[ptn_core] = pattern_count.get(ptn_core,0)+1

        res = ptn_learn.sentence_to_pattern(sentencej, sentencei)
        if res and len(res)>5:
            ptn_core = ptn_learn.get_pattern_core(res)
            pattern_list.add(ptn_core)
            pattern_full_count.append(res)
            # print sentencei
            # print sentencej
            # print ptn_core
            # print "======"
            # pattern_count[ptn_core] = pattern_count.get(ptn_core,0)+1
        # time.sleep(1)

    for ptn in pattern_list:
        pattern_count[ptn] = pattern_count.get(ptn_core, 0)+1
sort_ptn = sorted(pattern_count.items(), key=lambda x:x[1], reverse=True)
for idx,item in enumerate(sort_ptn):
    print item[0], item[1]
    # ptn_learn.pattern_mergeing(item)
'''
'''
new_ptn_count = {}
for idx, ptn in enumerate(sort_ptn):
    w1 = int(ptn[1])
    for j in range(idx+1, len(sort_ptn)):
        next_ptn = sort_ptn[j][0]
        w2 = int(sort_ptn[j][1])
        new_ptn = ptn_learn.pattern_mergeing(ptn[0].split(","),1,next_ptn.split(","),2)

        if new_ptn:
            pt = ",".join(new_ptn)
            new_ptn_count[pt] = new_ptn_count.get(pt,0)+w1+w2
        else:
            # new_ptn_count[ptn[0]] = w1
            # new_ptn_count[next_ptn[0]] = w2
            pass
new_ptn_sort_count = sorted(new_ptn_count.items(), key=lambda x:x[1], reverse=True)
for item in new_ptn_sort_count:
    print item[0], item[1]
'''

'''
a="ASR,*,ASE,提供,*,MEY,*,担保,。"
b="ASR,*,ASE,*,提供,MEY,担保,额度,。"
print a
print b
res = ptn_learn.pattern_mergeing(a.split(","),1,b.split(","),2)

print " ".join(res)
'''
