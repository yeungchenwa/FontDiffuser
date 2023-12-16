import argparse
import gradio as gr
from sample import (arg_parse, 
                    sampling,
                    load_fontdiffuer_pipeline)


def run_fontdiffuer(source_image, 
                    character, 
                    reference_image,
                    sampling_step,
                    guidance_scale,
                    batch_size):
    args.character_input = False if source_image is not None else True
    args.content_character = character
    args.sampling_step = sampling_step
    args.guidance_scale = guidance_scale
    args.batch_size = batch_size
    out_image = sampling(
        args=args,
        pipe=pipe,
        content_image=source_image,
        style_image=reference_image)
    return out_image


if __name__ == '__main__':
    args = arg_parse()
    args.demo = True
    args.ckpt_dir = 'ckpt'
    args.ttf_path = 'ttf/KaiXinSongA.ttf'

    # load fontdiffuer pipeline
    pipe = load_fontdiffuer_pipeline(args=args)

    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column(scale=1):
                gr.HTML("""
                    <div style="text-align: center; max-width: 1200px; margin: 20px auto;">
                    <h1 style="font-weight: 900; font-size: 3rem; margin: 0rem">
                        FontDiffuser
                    </h1>
                    <h2 style="font-weight: 450; font-size: 1rem; margin: 0rem">
                        <a href="https://yeungchenwa.github.io/"">Zhenhua Yang</a>, 
                        <a href="https://scholar.google.com/citations?user=6zNgcjAAAAAJ&hl=zh-CN&oi=ao"">Dezhi Peng</a>, 
                        Yuxin Kong, Yuyi Zhang, 
                        <a href="https://scholar.google.com/citations?user=IpmnLFcAAAAJ&hl=zh-CN&oi=ao"">Cong Yao</a>, 
                        <a href="http://www.dlvc-lab.net/lianwen/Index.html"">Lianwen Jin</a>†
                    </h2>
                    <h2 style="font-weight: 450; font-size: 1rem; margin: 0rem">
                        <strong>South China University of Technology</strong>, Alibaba DAMO Academy
                    </h2>
                    <h3 style="font-weight: 450; font-size: 1rem; margin: 0rem"> 
                    [<a href="https://github.com/yeungchenwa/FontDiffuser" style="color:blue;">arXiv</a>] 
                    [<a href="https://github.com/yeungchenwa/FontDiffuser" style="color:green;">Github</a>]
                    </h3>
                    <h2 style="text-align: left; font-weight: 600; font-size: 1rem; margin-top: 0.5rem; margin-bottom: 0.5rem">
                    1.We propose FontDiffuser, which is capable to generate unseen characters and styles, and it can be extended to the cross-lingual generation, such as Chinese to Korean.
                    </h2>
                    <h2 style="text-align: left; font-weight: 600; font-size: 1rem; margin-top: 0.5rem; margin-bottom: 0.5rem">
                    2. FontDiffuser excels in generating complex character and handling large style variation. And it achieves state-of-the-art performance.
                    </h2>
                    </div>
                    """)
                gr.Image('figures/result_vis.png')
                gr.Image('figures/demo_pipeline.png')
            with gr.Column(scale=1):
                with gr.Row():
                    with gr.Column():
                        source_image = gr.Image(width=320, label='[Option 1] Source Image', image_mode='RGB', type='pil')
                        character = gr.Textbox(value='隆', label='[Option 2] Source Character')
                        reference_image = gr.Image(width=320, label='Reference Image', image_mode='RGB', type='pil')
                    output_image = gr.Image(width=320, label="Output Image", image_mode='RGB', type='pil')

                sampling_step = gr.Slider(20, 50, value=20, step=10, 
                                          label="Sampling Step", info="The sampling step by FontDiffuser.")
                guidance_scale = gr.Slider(1, 12, value=7.5, step=0.5, 
                                           label="Scale of Classifier-free Guidance", 
                                           info="The scale used for classifier-free guidance sampling")
                batch_size = gr.Slider(1, 4, value=1, step=1, 
                                       label="Batch Size", info="The number of images to be sampled.")

                FontDiffuser = gr.Button('Run FontDiffuser')
                gr.Markdown("## <font color=#008000, size=6>Examples that You Can Choose Below⬇️</font>")
        with gr.Row():
            gr.Markdown("## Examples")
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## Example 1️⃣: Source Image and Reference Image")
                gr.Markdown("### In this mode, we provide both the source image and \
                            the reference image for you to try our demo!")
                gr.Examples(
                    examples=[['figures/source_imgs/source_灨.jpg', 'figures/ref_imgs/ref_籍.jpg'], 
                            ['figures/source_imgs/source_鑫.jpg', 'figures/ref_imgs/ref_簸.jpg']],
                    inputs=[source_image, reference_image]
                )
            with gr.Column(scale=1):
                gr.Markdown("## Example 2️⃣: Character and Reference Image")
                gr.Markdown("### In this mode, we provide the content character and the reference image \
                            for you to try our demo!")
                gr.Examples(
                    examples=[['霸', 'figures/ref_imgs/ref_籍.jpg'], 
                            ['窿', 'figures/ref_imgs/ref_簸.jpg']],
                    inputs=[character, reference_image]
                )
            with gr.Column(scale=1):
                gr.Markdown("## Example 3️⃣: Reference Image")
                gr.Markdown("### In this mode, we provide only the reference image, \
                            you can upload your own source image or you choose the character above \
                            to try our demo!")
                gr.Examples(
                    examples=['figures/ref_imgs/ref_籍.jpg', 
                            'figures/ref_imgs/ref_簸.jpg'],
                    inputs=reference_image
                )
        FontDiffuser.click(
            fn=run_fontdiffuer,
            inputs=[source_image, 
                    character, 
                    reference_image,
                    sampling_step,
                    guidance_scale,
                    batch_size],
            outputs=output_image)
    demo.launch(debug=True)