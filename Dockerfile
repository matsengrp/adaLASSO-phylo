FROM matsen/phyloinfer

COPY . /adalasso
WORKDIR /adalasso

RUN ln -s /adalasso/bin/linux/raxml-ng /usr/bin

RUN /opt/conda/bin/conda run -n phyloinfer jupyter nbconvert --to notebook --inplace --execute --ExecutePreprocessor.timeout=10000 examples/denv4-brazil.ipynb
RUN /opt/conda/bin/conda run -n phyloinfer jupyter nbconvert --to notebook --inplace --execute --ExecutePreprocessor.timeout=10000 examples/FISTA-vs-ISTA.ipynb
RUN /opt/conda/bin/conda run -n phyloinfer jupyter nbconvert --to notebook --inplace --execute --ExecutePreprocessor.timeout=10000 examples/short-edge-support.ipynb
RUN /opt/conda/bin/conda run -n phyloinfer jupyter nbconvert --to notebook --inplace --execute --ExecutePreprocessor.timeout=10000 examples/simulated-100tips.ipynb
