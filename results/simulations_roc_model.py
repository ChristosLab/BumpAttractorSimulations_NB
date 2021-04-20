# -*- coding: utf-8 -*-
"""
@author: David Bestue
"""

from model import *
numcores = multiprocessing.cpu_count() - 3


N=512
ch_size = 1
ch = int(360/ch_size)
reps_=100


Targets = list(np.arange(0, 360, ch_size))* reps_


path_save_re_ON= '/home/david/Desktop/Christos_neuron/simuls_on_class_re.xlsx'
path_save_targets_ON= '/home/david/Desktop/Christos_neuron/simuls_on_class_targets.xlsx'


print('Simulations ON')

outputs_ON= Parallel(n_jobs = numcores)(delayed(model)(totalTime=1000, targ_onset_1=0, targ_onset_2=4000, angle_target_i=iPos, presentation_period=100, 
    angle_separation=170, tauE=20, tauI=10,  n_stims=2, I0E=0.05, I0I=0.5, 
    GEE=0.068, 
    GII= 0.13,
    GEI=0.13,
    GIE=0.042, 
    sigE=7., sigI=5., k_noise=0.6,           
    kappa_E=45, 
    kappa_I=0.3, 
    kappa_stim=40., N=512, stim_strengthE=9.4, stim_strengthI=0.,
    plot_connectivity=False, plot_rate=False, plot_hm=False , plot_fit=False, 
    phantom_st=1.2, phantom_onset=50000, phnatom_duration=100, just_final_re=True) for iPos in Targets) 



X_on = np.array(outputs_ON).reshape(ch*reps_, N)
y_on = np.array(Targets)


pd.DataFrame(X_on).to_excel(path_save_re_ON) 
pd.DataFrame(y_on).to_excel(path_save_targets_ON) 


##############################################################################################################################################
##############################################################################################################################################
##############################################################################################################################################


print('Simulations OFF')



path_save_re_OFF= '/home/david/Desktop/Christos_neuron/simuls_off_class_re.xlsx'
path_save_targets_OFF= '/home/david/Desktop/Christos_neuron/simuls_off_class_targets.xlsx'


outputs_OFF= Parallel(n_jobs = numcores)(delayed(model)(totalTime=1000, targ_onset_1=0, targ_onset_2=4000, angle_target_i=iPos, presentation_period=100, 
    angle_separation=170, tauE=20, tauI=10,  n_stims=2, I0E=-3.5, I0I=0.5, 
    GEE=0.068, 
    GII= 0.13,
    GEI=0.13,
    GIE=0.042, 
    sigE=7., sigI=5., k_noise=0.6,           
    kappa_E=45, 
    kappa_I=0.3, 
    kappa_stim=40., N=512, stim_strengthE=9.4, stim_strengthI=0.,
    plot_connectivity=False, plot_rate=False, plot_hm=False , plot_fit=False, 
    phantom_st=1.2, phantom_onset=50000, phnatom_duration=100, just_final_re=True) for iPos in Targets) 



X_off = np.array(outputs_OFF).reshape(ch*reps_, N)
y_off = np.array(Targets)

pd.DataFrame(X_off).to_excel(path_save_re_OFF) 
pd.DataFrame(y_off).to_excel(path_save_targets_OFF) 




# import numpy as np
# import matplotlib.pyplot as plt
# from itertools import cycle
# from sklearn import svm, datasets
# from sklearn.metrics import roc_curve, auc
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import label_binarize
# from sklearn.multiclass import OneVsRestClassifier
# from scipy import interp
# from sklearn.metrics import roc_auc_score

# X=X_on
# y=y_on
# #####np.random.shuffle(y)


# # Binarize the output
# y= label_binarize(y, classes=np.arange(0, 360,ch_size))
# n_classes = y.shape[1]

# random_state = np.random.RandomState(0)
# n_samples, n_features = X.shape


# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5, random_state=0)

# classifier = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True, random_state=random_state))

# y_score = classifier.fit(X_train, y_train).decision_function(X_test)


# # Compute ROC curve and ROC area for each class
# fpr = dict()
# tpr = dict()
# roc_auc = dict()
# for i in range(n_classes):
#     fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
#     roc_auc[i] = auc(fpr[i], tpr[i])

# # Compute micro-average ROC curve and ROC area
# fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
# roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])


# plt.figure()
# lw = 2
# plt.plot(fpr[2], tpr[2], color='darkorange',
#          lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
# plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
# # plt.xlim([0.0, 1.0])
# # plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver operating characteristic example')
# plt.legend(loc="lower right", fontsize=15)
# plt.show()









