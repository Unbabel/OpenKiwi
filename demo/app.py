from pathlib import Path

import streamlit as st


@st.cache(allow_output_mutation=True)
def read_lines(path):
    return [line.strip() for line in open(path)]


def main():

    # TODO: both don't work...
    # st.sidebar.image("demo/assets/img/openkiwi-logo-horizontal.svg")
    # st.sidebar.image("demo/assets/img/logo.ico")
    st.sidebar.title("Kiwi Tasting")
    st.sidebar.markdown("Inspect predictions by OpenKiwi.")
    st.sidebar.markdown(
        "[paper](https://www.aclweb.org/anthology/P19-3020/) | "
        "[code](https://www.aclweb.org/anthology/P19-3020/)"
    )
    st.sidebar.markdown("---")

    models = {}

    # st.sidebar.header("Model")
    # model_path = st.sidebar.text_input("Path")
    # if model_path:
    #     model_path = Path(model_path)
    #     if not model_path.exists():
    #         raise ValueError(f"model path does not exist: {model_path}")
    #     else:
    #         load_aligner(model_path, matching_methods)
    #
    # models = {}
    # mode[model_path.name] = load_aligner(model_path, matching_methods)
    #
    # if second_model:
    #     aligners[second_model] = load_aligner(second_model, matching_methods)
    # elif second_model_path:
    #     aligners[second_model_path.name] = load_aligner(second_model_path, matching_methods)

    st.header("Input")
    st.write("Specify paths to input data.")
    source_path = st.text_input("Source")
    target_path = st.text_input("Target")
    tags_path = st.text_input("Tags")
    probs_path = st.text_input("Probabilities")
    scores_path = st.text_input("Scores")

    lines = {}
    if source_path and Path(source_path).exists():
        lines['source'] = read_lines(source_path)
    if target_path and Path(target_path).exists():
        lines['target'] = read_lines(target_path)
    if tags_path and Path(tags_path).exists():
        lines['tags'] = read_lines(tags_path)
    if probs_path and Path(probs_path).exists():
        lines['probs'] = read_lines(probs_path)
    if scores_path and Path(scores_path).exists():
        lines['scores'] = read_lines(scores_path)

    if 'source' in lines and 'target' in lines and ('tags' in lines or 'probs' in lines or 'scores' in lines):
        st.markdown("---")
        i = lines['source'].index(
            st.selectbox("Choose source line to display", options=lines['source'])
        )
        source = lines['source'][i].split()
        target = lines['target'][i].split()
        
        st.write(' '.join(source))
        st.write(' '.join(target))
        if 'tags' in lines:
            st.write(lines['tags'][i])
        if 'probs' in lines:
            st.write(lines['probs'][i])
        if 'scores' in lines:
            st.write(lines['scores'][i])


if __name__ == "__main__":
    main()
