from maskgen_loader import MaskGenLoader
import plugins
from software_loader import getOperations, Operation, getOperation, getOperationsByCategory, insertCustomRule, getRule,getFilters
import logging

maskgenloader = MaskGenLoader()


def callRule(functions, *args, **kwargs):
    import copy
    for func in functions:
        edge = args[0]
        edgeMask = args[1]
        res = getRule(func)(edge, edgeMask, **kwargs)
        kwargs = copy.copy(kwargs)
        if 'donorMask' in kwargs and 'donorMask' is not None:
            kwargs['donorMask'] = res
        else:
            kwargs['compositeMask'] = res
    return res


def addToSet(aSet, aList):
    for item in aList:
        if item not in aSet:
            aSet.add(item)


def addToMap(to_map, from_map):
    for k, v in from_map.iteritems():
        to_map[k] = v


def get_transitions(operations):
    if len(operations) == 0:
        return []
    first_op = operations[0]
    second_op = operations[-1]
    transitions = []
    for first_op_trans in first_op.transitions:
        start = first_op_trans.split('.')[0]
        for second_op_trans in second_op.transitions:
            end = second_op_trans.split('.')[1]
            transition = start + '.' + end
            if transition not in transitions:
                transitions.append(transition)
    return transitions

def buildFilterOperation(pluginOp):
    import copy
    if pluginOp is None:
        return None
    realOp = getOperation(pluginOp['name'],fake=True)
    mandatory = copy.copy(realOp.mandatoryparameters)
    for k,v in  (pluginOp['arguments']  if 'arguments' in pluginOp and pluginOp['arguments'] is not None else {}).iteritems():
        mandatory[k] = v
    optional = {k:v for k,v in realOp.optionalparameters.iteritems() if k not in mandatory}
    return Operation(name=pluginOp['name'],
                     category=pluginOp['category'],
                     generateMask=realOp.generateMask,
                     mandatoryparameters=mandatory,
                     description=pluginOp['description'],
                     optionalparameters=optional,
                     rules=realOp.rules,
                     includeInMask=realOp.includeInMask,
                     transitions=pluginOp['transitions'])

class GroupFilter:
    name = None
    filters = []

    def __init__(self, name, filters):
        self.name = name
        self.filters = filters

    def isValid(self):
        for filter in self.filters:
            if plugins.getOperation(filter) is None:
                logging.getLogger('maskgen').warning('Invalid filter {} in group {}'.format( filter, self.name))
                return False
        return True

    def getOperation(self):
        op = {}
        op['arguments'] = {}
        bestsuffix = None
        ops = []
        for filter in self.filters:
            suffix = plugins.getPreferredSuffix(filter)
            if suffix is not None:
                bestsuffix = suffix
            currentop = plugins.getOperation(filter)
            ops.append(buildFilterOperation(currentop))
            for k,v in currentop.iteritems():
                if k == 'arguments' and v is not None:
                    for arg,definition in v.iteritems():
                        if arg not in op['arguments']:
                            op['arguments'][arg] = definition
                elif k not in op:
                    op[k] = v
        op['transitions'] = get_transitions(ops)
        return {'function':'custom',
                'operation':op,
                'command':'custom',
                'suffix': bestsuffix,
                'mapping':None,
                'group':'Sequence'
                }

class OperationGroupFilter(GroupFilter):

    def __init__(self, name, filters):
        GroupFilter.__init__(self,name,filters)

    def isValid(self):
        for filter in self.filters:
            if getOperation(filter) is None:
                logging.getLogger('maskgen').warning('Invalid operation {} in group {}'.format(filter, self.name))
                return False
        return True

class GroupFilterLoader:
    groups = {}

    def getAvailableFilters(self, operations_used=list()):
        p = plugins.getOperations()
        names = p.keys()

        # grab transition prefixes for last operation
        transitionsPre = [t.split('.')[0] for t in
                          p[operations_used[-1]]['operation']['transitions']] if operations_used else None

        result = []
        for op_name in names:
            if op_name not in operations_used:
                if transitionsPre is not None:
                    # grab transition prefixes for current op
                    op_transitions = [t.split('.')[0] for t in p[op_name]['operation']['transitions']]

                    # don't append and continue with loop if transitions don't match
                    if set(transitionsPre).isdisjoint(op_transitions):
                        continue
                result.append(op_name)
        return result

    def getLoaderKey(self):
        return "filtergroups"

    def getGroups(self):
        return self.groups.values()

    def getGroup(self, name):
        return self.groups[name] if name in self.groups else None

    def getGroupNames(self):
        return self.groups.keys()

    def _getOperation(self,name):
        pluginOp =  plugins.getOperation(name)
        return buildFilterOperation(pluginOp)

    def _buildGroupOperation(self,grp, name):
        from functools import partial
        if grp is not None:
            includeInMask = False
            rules = set()
            opt_params = dict()
            mandatory_params = dict()
            analysisOperations = set()
            generateMask = False
            grp_categories = set()
            customFunctions = []
            ops = []
            if not grp.isValid():
                return None
            for op in grp.filters:
                operation = self._getOperation(op)
                ops.append(operation)
                grp_categories.add(operation.category)
                includeInMask |= operation.includeInMask
                generateMask |= operation.generateMask
                addToSet(rules, operation.rules)
                addToMap(mandatory_params, operation.mandatoryparameters)
                addToMap(opt_params, operation.optionalparameters)
                addToSet(analysisOperations, operation.analysisOperations)
                if operation.maskTransformFunction is not None:
                    customFunctions.append(operation.maskTransformFunction)
                else:
                    customFunctions.append("maskgen.mask_rules.defaultMaskTransform")
            opt_params = dict([(k, v) for (k, v) in opt_params.iteritems() if k is not mandatory_params])
            transitions = get_transitions(ops)

            maskTransformFunction = None
            if len(customFunctions) > 0:
                maskTransformFunction = name + '_mtf'
                insertCustomRule(maskTransformFunction, partial(callRule, customFunctions))
            return Operation(name=name, category='Groups',
                             includeInMask=includeInMask,
                             generateMask=generateMask,
                             mandatoryparameters=mandatory_params,
                             description='User Defined',
                             optionalparameters=opt_params,
                             rules=list(rules),
                             transitions=transitions,
                             groupedOperations=grp.filters,
                             groupedCategories=grp_categories,
                             analysisOperations=analysisOperations,
                             maskTransformFunction=maskTransformFunction)
        return getOperation(name,fake=True)

    def getOperation(self, name):
        grp = self.getGroup(name)
        return self._buildGroupOperation(grp, name)

    def __init__(self):
        self.load()

    def getOperations(self,startType,endType):
        import copy
        p = copy.copy(plugins.getOperations(startType))
        for grp,v in self.groups.iteritems():
            if not v.isValid():
                continue
            grpOp = v.getOperation()
            transitions = [t.split('.')[0] for t in grpOp['operation']['transitions']]
            if startType is None or startType in transitions:
                p[grp] =grpOp
        return p

    def load(self, filterFactory=lambda k,v: GroupFilter(k, v)):
        global maskgenloader
        plugins.loadPlugins()
        self.groups = {}
        newset = maskgenloader.get_key(self.getLoaderKey())
        if newset is not None:
            for k, v in newset.iteritems():
                if len(v) > 0:
                    group = filterFactory(k, v)
                    if group.isValid():
                        self.groups[k] = group
        for k,v in getFilters(self.getLoaderKey()).iteritems():
            if len(v) > 0:
                group = filterFactory(k, v)
                if group.isValid():
                    self.groups[k] = group

    def add(self, groupfilter):
        self.groups[groupfilter.name] = groupfilter

    def remove(self, name):
        self.groups.pop(name)

    def getName(self):
        return "Filters"

    def save(self):
        global maskgenloader
        image = {}
        for k, v in self.groups.iteritems():
            image[k] = v.filters
        maskgenloader.save(self.getLoaderKey(), image)





"""
Rules needed:
   Only one generateMask allowed
   Transitions must be checked to be valid
"""


class GroupOperationsLoader(GroupFilterLoader):
    def getLoaderKey(self):
        return "operationsgroups"

    def getName(self):
        return "Operations"

    def __init__(self):
        GroupFilterLoader.load(self,filterFactory=lambda k,v: OperationGroupFilter(k,v))

    def _getOperation(self, name):
        return  getOperation(name, fake=True)

    def getAvailableFilters(self, operations_used=list()):
        has_generate_mask = False
        for op_name in operations_used:
            real_op = getOperation(op_name, fake=True)
            has_generate_mask |= real_op.generateMask
        ops = getOperations()
        return [op_name for op_name in ops if op_name not in operations_used and not \
            (has_generate_mask and ops[op_name].generateMask)]

    def getCategoryForGroup(self, groupName):
        if groupName in self.getGroupNames():
            return "Groups"
        return None

    def getOperations(self,fileType):
        import copy
        p = copy.copy(plugins.getOperations(fileType))
        for grp,v in self.groups.iteritems():
            p[grp] = v.getOperation()
        return p

    def getOperations(self, startType, endType):
        cat = dict()
        cat['Groups'] = list()
        newset = maskgenloader.get_key(self.getLoaderKey())
        if newset is None:
            return cat
        deleteKeys  = []
        for group, content in newset.iteritems():
            if len(content) > 0:
                first_op = getOperation(content[0])
                if first_op is None:
                    deleteKeys.append(group)
                    continue
                for first_op_trans in first_op.transitions:
                    start = first_op_trans.split('.')[0]
                    if start != startType:
                        continue
                    second_op = getOperation(content[-1])
                    if second_op is None:
                        deleteKeys.append( group)
                        continue
                    for second_op_trans in second_op.transitions:
                        end = second_op_trans.split('.')[1]
                        if end == endType:
                            cat['Groups'].append(group)
                            continue
        for group in deleteKeys:
            newset.pop(group)
        return cat


groupOpLoader = GroupOperationsLoader()


def injectGroup(name, ops):
    global groupOpLoader
    groupOpLoader.groups[name] = OperationGroupFilter(name, ops)

def getCategoryForOperation(name):
    global groupOpLoader
    ops = getOperations()
    if name in ops:
        return ops[name].category
    return groupOpLoader.getCategoryForGroup(name)


def getOperationWithGroups(name, fake=False, warning=True):
    global groupOpLoader
    op = getOperation(name, fake=False, warning=False)
    if op is None:
        op = groupOpLoader.getOperation(name)
    if op is None and fake:
        return getOperation(name, fake=True, warning=warning)
    return op


def getOperationsByCategoryWithGroups(sourcetype, targettype):
    global groupOpLoader
    res = dict(getOperationsByCategory(sourcetype, targettype))
    items = groupOpLoader.getOperations(sourcetype, targettype)
    if items is not None:
        for k, v in items.iteritems():
            res[k] = v
    return res
