from collections import defaultdict
import utils


class TypeHierarchy:
    def __init__(self, file_name, number_of_types):
        self._type_hierarchy = {}  # type -> [parent type]
        self._subtype_mapping = defaultdict(list)  # type -> [subtype]
        self._root = set()  # root types (on 1-level)
        with open(file_name) as f:
            for line in f:
                t = line.strip('\n').split('\t')
                self._type_hierarchy[int(t[0])] = int(t[1])
                self._subtype_mapping[int(t[1])].append(int(t[0]))
                self._root.add(int(t[0]))
        self._root = list(set(range(1, number_of_types+1)).difference(self._root))

    def get_type_path(self, label):
        if label in self._type_hierarchy:  # label has super type
            path = [label]
            while label in self._type_hierarchy:
                path.append(self._type_hierarchy[label])
                label = self._type_hierarchy[label]
            path.reverse()
            return path
        else:  # label is the root type
            return [label]

    def get_subtypes(self, label):
        if label in self._subtype_mapping:
            return self._subtype_mapping[label]
        else:
            return None

def get_short2full_map(require_type_lst):
    short2full = {}
    for full in require_type_lst:
        short = full.split('/')[-1]
        short2full[short] = full
    return short2full




def refine(labels,tier,maxDepth=3):
    keep = [""] * maxDepth
    short2full = get_short2full_map(utils.get_ontoNotes_train_types())
    for l in labels:
        path = l.split('/')[1:]
        path = [short2full[k] for k in path]
        for i in range(len(path)):
            if keep[i] == "":
                keep[i] = path[i]
            elif keep[i] != path[i]:
                break

    return [l for l in keep if l != ""]



if __name__ == '__main__':
    th = TypeHierarchy('type_h.txt', 3)
    print(th.get_type_path(2))
    pass
    print('/per/art'.split('/'))
    a = [1,0,1,0]
    print(a.index(1))

