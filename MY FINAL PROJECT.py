import streamlit as st
from PIL import Image

page_bg_img = """
<style> 
[data-testid="stAppViewContainer"]{
background-image: url(https://img.freepik.com/free-vector/gradient-dark-dynamic-lines-background_23-2148995950.jpg?w=996&t=st=1684305227~exp=1684305827~hmac=5e1a40572f7298508c1dfd733bf68a029c30ce3a0ebc79e6db33c4f2c740ba12);
background-size: cover;}
[data-testid = "stHeader"]{
background-color: rgba(0,0,0,0);
}
</style>
"""

st.set_page_config(
    page_title="BÁO CÁO CUỐI KÌ",
    page_icon="▶",
)


#lottie_coding = load_lottieur1("https://assets5.lottiefiles.com/packages/lf20_4kx2q32n.json")
img_contact_form = Image.open("images/huy1.png")


# Define CSS styling for the header text
header_style = """
    <style>
        .header-text {
            color: white;
        }
    </style>
"""

# Display the header text with the defined CSS styling
st.markdown(header_style, unsafe_allow_html=True)
st.markdown('<h1 class="header-text">BÀI BÁO CÁO CUỐI KÌ</h1>', unsafe_allow_html=True)

with st.container():
    st.markdown(page_bg_img, unsafe_allow_html=True)

with st.container(): 
    st.write("---")
    text_column, image_column = st.columns((4, 2))
    with text_column:
        st.markdown(
            """
            ### Môn học: Học Máy
            ### GVHD: Ts. Trần Tiến Đức
            ### SVTH: Tăng Hoàng Huy
            ### MSSV: 20146342
            """
        )
    with image_column:
        st.image(img_contact_form)

