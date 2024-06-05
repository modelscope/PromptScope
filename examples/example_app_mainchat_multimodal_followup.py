from meta_icl.contribs.app_main_followup.app_main_followup_str import get_followup_results
from meta_icl.utils.sys_prompt_utils import load_json_file

if __name__ == '__main__':
    config_pth = "conf/app_followup_configs/app_followup_multimodal_conf.json"
    conf = load_json_file(config_pth)
    file_type="image"
    cur_query = {
        "chat_history":
            [
                {"role": "user", "content": "我该如何回复？"},
                {
                    "role": "assistant",
                    "content": "这是一张显示社交媒体界面的图片，上面有用户发表的文字和评论。如果你想回复某个评论，通常需要点击评论下方的回复按钮或者直接在输入框中输入你的回复内容，然后点击发送按钮将你的回复发布出去。如果你想要对整个帖子进行回复，可以在帖子下方找到相应的回复按钮并按照同样的步骤操作。"
                },
                {"role": "user", "content": "我该如何回复？"},
                {"role": "assistant", "content": "根据您提供的信息，这是一条社交媒体上的帖子。如果您想在该平台上回复这条帖子，您可以点击“发表评论”按钮，然后输入您的回复内容，最后点击发送即可。在这里，由于这是一个公开的平台，建议您注意言辞得体，并尊重其他用户的观点。"}
             ],
        "last_query": "我该如何回复？",
        "last_answer": "这张图片显示的是一个社交媒体应用的截图，上面有用户发表的文字和评论。文字内容似乎在描述射手座的理想型，并且附带了一首背景音乐的名字《琉璃月·品味人生》。\n\n下面是几条用户的评论：\n\n- 涵：实在不行你当我男朋友得了。\n- 涵：你又懂了。\n- 雪莲：瞅着可以那就滚出我的视线线！\n- 射手座&文案：发表评论（这里有一个笑脸表情符号）。\n\n如果你想要回复这些评论，你可以点击每个评论下方的“发表评论”按钮并输入你的回应。例如，如果你想回复涵的第一条评论，你可以写：“哈哈，谢谢你的提议，我们再看看吧。”然后点击发送即可。记得保持礼貌和尊重，即使是在网络上交流也要注意言辞得体。"
    }

    results = get_followup_results(cur_query, embedding_key=conf["icl_configs"]["embedding_key"],
                                   base_model=conf["task_configs"]["base_model"],
                                   embedding_pth=conf["icl_configs"]["embedding_pth"],
                                   examples_pth=conf["icl_configs"]["examples_pth"],
                                   embedding_model=conf["icl_configs"]["embedding_model"],
                                   model_config=None,
                                   task_config=conf["task_configs"],
                                   num=conf["icl_configs"]["topk"],
                                   file_type=file_type)
    print(results)
