import matplotlib.pyplot as plt
import numpy as np

losses = [1.6770522594451904, 0.6683252453804016, 0.4533480405807495, 0.36815306544303894, 0.19559001922607422,
          0.10680298507213593, 0.08850124478340149, 0.08494430780410767, 0.09913311153650284, 0.04614133760333061,
          0.09281452000141144, 0.045382630079984665, 0.08557461947202682, 0.06450577080249786, 0.028975367546081543,
          0.018162516877055168, 0.01646229811012745, 0.02107628434896469, 0.011119704693555832, 0.01640656404197216,
          0.011804102919995785, 0.016521930694580078, 0.010649976320564747, 0.008434475399553776, 0.01000448688864708,
          0.008350321091711521, 0.017060501500964165, 0.010218368843197823, 0.01912843994796276, 0.009715057909488678]

loses2 = [0.1870070993900299, 0.16179266571998596, 0.15131983160972595, 0.032329902052879333, 0.08536683768033981, 0.002387527609243989, 0.005761221516877413, 0.0029576809611171484, 0.002801136579364538, 0.000697254145052284, 0.0008556079119443893, 0.000658096803817898, 0.0005985555471852422, 0.000896797573659569, 0.0013700290583074093, 0.00043009716318920255, 0.404875785112381, 0.0029568574391305447, 0.002701923716813326, 0.0004951476585119963, 0.0003582400386221707, 0.0013802798930555582, 0.001467516995035112, 0.0007555951015092432, 0.002056536264717579, 0.0015975971473380923, 0.0075813159346580505, 0.0008054742938838899, 0.0003165378875564784, 0.0005032707122154534]

plt.plot(np.array(list(range(len(losses)))),losses)
plt.xlabel("steps")
plt.ylabel("loss")
plt.title("LOSS 0.001 and LOSS 0.01")

plt.plot(np.array(list(range(len(loses2)))),loses2)
plt.show()