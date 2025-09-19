import os
import argparse
from experiment_helpers.gpu_details import print_details
from accelerate import Accelerator
import time
import torch
import numpy as np
from single_object import ObjectRFlow
from single_person import PersonRFlow
import wandb
from transformers import AutoProcessor, CLIPModel
#import ImageReward as RM
from style_rl.image_utils import concat_images_horizontally
from style_rl.eval_helpers import DinoMetric
from style_rl.prompt_list import real_test_prompt_list
from datasets import load_dataset,Dataset

parser=argparse.ArgumentParser()

parser.add_argument("--mixed_precision",type=str,default="no")
parser.add_argument("--model",type=str,default="person",help="person or object")
parser.add_argument("--src_dataset",type=str, default="jlbaker361/mtg")
parser.add_argument("--num_inference_steps",type=int,default=20)
parser.add_argument("--project_name",type=str,default="baseline")
parser.add_argument("--limit",type=int,default=-1)
parser.add_argument("--size",type=int,default=512)


@torch.no_grad()
def main(args):
    #ir_model=RM.load("ImageReward-v1.0")
        
        
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
    accelerator=Accelerator(log_with="wandb",mixed_precision=args.mixed_precision)
    accelerator.init_trackers(project_name=args.project_name,config=vars(args))

    dino_metric=DinoMetric(accelerator.device)

    torch_dtype={
        "no":torch.float32,
        "fp16":torch.float16,
        "bf16":torch.bfloat16
    }[args.mixed_precision]
    device=accelerator.device

    if args.model=="person":
        model_class=PersonRFlow
    elif args.model=="object":
        model_clsss=ObjectRFlow

    model_rflow=model_class('PeRFlow',123,device,torch_dtype,args.size)

    data=load_dataset(args.src_dataset, split="train")

    background_data=load_dataset("jlbaker361/real_test_prompt_list",split="train")
    background_dict={row["prompt"]:row["image"] for row in background_data}

    text_score_list=[]
    image_score_list=[]
    image_score_background_list=[]
    #ir_score_list=[]
    dino_score_list=[]


    for k,row in enumerate(data):
        if k==args.limit:
            break
        
        prompt=real_test_prompt_list[k%len(real_test_prompt_list)]
        background_image=background_dict[prompt]
        image=row["image"]

        model_rflow.prepare(image,"person")
        augmented_image=model_rflow.generate(prompt,123,verbose=False)


        concat=concat_images_horizontally([image,augmented_image])

        accelerator.log({
            f"image_{k}":wandb.Image(concat)
        })

        inputs = processor(
                text=[prompt], images=[image,augmented_image,background_image], return_tensors="pt", padding=True
        )

        outputs = clip_model(**inputs)
        image_embeds=outputs.image_embeds
        text_embeds=outputs.text_embeds
        logits_per_text=torch.matmul(text_embeds, image_embeds.t())[0]
        #accelerator.print("logits",logits_per_text.size())

        image_similarities=torch.matmul(image_embeds,image_embeds.t()).numpy()[0]

        [_,text_score,__]=logits_per_text
        [_,image_score,image_score_background]=image_similarities
        #ir_score=ir_model.score(prompt,augmented_image)
        dino_score=dino_metric.get_scores(image, [augmented_image])

        text_score_list.append(text_score.detach().cpu().numpy())
        image_score_list.append(image_score)
        image_score_background_list.append(image_score_background)
        #ir_score_list.append(ir_score)
        dino_score_list.append(dino_score)

    accelerator.log({
        "text_score_list":np.mean(text_score_list),
        "image_score_list":np.mean(image_score_list),
        "image_score_background_list":np.mean(image_score_background_list),
        #"ir_score_list":np.mean(ir_score_list),
        "dino_score_list":np.mean(dino_score_list)
    })



if __name__=='__main__':
    print_details()
    start=time.time()
    args=parser.parse_args()
    print(args)
    main(args)
    end=time.time()
    seconds=end-start
    hours=seconds/(60*60)
    print(f"successful generating:) time elapsed: {seconds} seconds = {hours} hours")
    print("all done!")