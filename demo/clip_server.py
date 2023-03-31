import os
import numpy as np
import clip
import argparse
import socket

def get_parser():
    parser = argparse.ArgumentParser(description='DisNet')
    parser.add_argument('--out_dir', type=str, default='./', help='specify the output directory')
    args = parser.parse_args()
    return args

def main():
    args = get_parser()
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    # "ViT-L/14@336px" # the big model that OpenSeg uses
    print('Loading the CLIP model...')
    clip_pretrained, _ = clip.load("ViT-L/14@336px", device='cpu', jit=False)
    ##clip_pretrained, _ = clip.load("ViT-L/14@336px", device='cuda', jit=False)
    print('Finished loading.')
    print('Ready for queries')

    HOST = "127.0.0.1"  # localhost
    PORT = 1111  # Port to listen on (non-privileged ports are > 1023)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
       s.bind((HOST, PORT))
       s.listen()
       conn, addr = s.accept()
       with conn:
           while True:
              query_buffer = conn.recv(1024)
              if not query_buffer:
                  break
              query_string = ""
              for c in query_buffer:
                  query_string +=  chr(c)
              print(query_string)

              # generate token
              text = clip.tokenize([query_string])
              #text = clip.tokenize([query_string]).cuda()
              text_features = clip_pretrained.encode_text(text)

              # # normalize
              text_features = text_features / text_features.norm(dim=-1, keepdim=True)
              #print(text_features)

              # save features
              np.save(os.path.join(out_dir, '{}.npy'.format(query_string)), text_features.detach().cpu().numpy())
              print('CLIP feature of "{}" is saved to {}'.format(query_string, os.path.join(out_dir, '{}.npy'.format(query_string))))

              # Send response
              conn.sendall(query_buffer)


if __name__ == '__main__':
    main()
