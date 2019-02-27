Using pre-trained models
========================

We provide the `models <https://github.com/Unbabel/OpenKiwi/releases/>`_ used in our submission to `Codalab <https://competitions.codalab.org/competitions/19306#results>`_ together with a script that ensembles their predictions over the dev set. For reference, these are our results (which you can reproduce by following the steps below)


+-----------+----------------------------------------+----------------------------------------+
|           |                En-De SMT               |                En-De NMT               |
+           +========================================+========================================+
|   Model   |                                        |                                        |
+           +-------+-------+--------+-------+-------+-------+-------+--------+-------+-------+
|           |   MT  |  gaps | source |   r   |   ⍴   |   MT  |  gaps | source |   r   |   ⍴   |
+-----------+-------+-------+--------+-------+-------+-------+-------+--------+-------+-------+
| QUETCH    | 39.90 | 17.10 |  36.10 | 48.32 | 51.31 | 29.18 | 13.26 |  28.91 | 42.84 | 49.59 |
+-----------+-------+-------+--------+-------+-------+-------+-------+--------+-------+-------+
| NuQE      | 50.04 | 35.53 |  42.08 | 59.62 | 60.89 | 32.49 | 15.01 |  30.19 | 43.41 | 50.87 |
+-----------+-------+-------+--------+-------+-------+-------+-------+--------+-------+-------+
| APE-QE    | 55.12 | 47.04 |  51.11 | 58.01 | 60.58 | 37.60 | 21.78 |  34.46 | 35.23 | 38.88 |
+-----------+-------+-------+--------+-------+-------+-------+-------+--------+-------+-------+
| Pred-Est  | 57.29 | 43.68 |  33.02 | 70.95 | 74.49 | 39.25 | 21.54 |  29.52 | 50.18 | 55.66 |
+-----------+-------+-------+--------+-------+-------+-------+-------+--------+-------+-------+
| Stacked   | 62.40 |       |        |       |       | 43.88 |       |        |       |       |
+-----------+-------+-------+--------+-------+-------+-------+-------+--------+-------+-------+
| Ensembled | 61.33 | 53.05 |  51.11 | 72.89 | 76.37 | 43.04 | 24.74 |  34.46 | 52.34 | 56.98 |
+-----------+-------+-------+--------+-------+-------+-------+-------+--------+-------+-------+

Reproducing benchmark values
---------------------------

Go to `WMT18 Download <https://lindat.mff.cuni.cz/repository/xmlui/handle/11372/LRT-2619>`_ and follow the instructions to download a zip archive with the development data. Then, open a shell and navigate to the directory where you downloaded the data. Ensure that OpenKiwi is installed in your environment, and run the following command(s) to download and evaluate our models:

(SMT dataset)::

    wget https://github.com/Unbabel/OpenKiwi/releases/download/0.1.1/en_de.smt_models.zip && unzip -n en_de.smt_models.zip && ./run_smt.sh

(NMT dataset)::

    wget https://github.com/Unbabel/OpenKiwi/releases/download/0.1.1/en_de.nmt_models.zip && unzip -n en_de.nmt_models.zip && ./run_nmt.sh
