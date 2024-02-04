import numpy as np
import time
from nltk.translate.bleu_score import corpus_bleu
from torch.nn.utils.rnn import pack_padded_sequence
from tqdm import tqdm
import config
from config import *


def evaluate(loader, encoder, decoder, word_map, max_len=100):
    """
    Evaluation
    :param data_folder: folder with data files saved by create_input_files.py
    :param data_name: base name shared by data files
    :return: BLEU-4 score
    """
    vocab_size = len(word_map)
    encoder.eval()
    decoder.eval()

    references = list()
    hypotheses = list()
    # reversed_word_map = {value: key for key, value in word_map.items()}

    with torch.no_grad():
        for i, (image, caps, caplens, allcaps) in enumerate(
                tqdm(loader, desc="EVALUATING..." )):

            image = image.to(device)                            # (batch_size, 3, 256, 256)

            # Encode
            encoder_out = encoder(image)                        # (batch_size, enc_image_size, enc_image_size, encoder_dim)
            encoder_dim = encoder_out.size(3)
            batch_size = encoder_out.size(0)

            # Flatten encoding
            encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # (batch_size, num_pixels, encoder_dim)

            # Initialize LSTM
            prev_word = torch.LongTensor([[word_map['<start>']]] * batch_size).to(device)   # (batch_size, 1)
            seqs = prev_word                                                                    # (batch_size, 1)
            h, c = decoder.init_hidden_state(encoder_out)                                        # (batch_size, decoder_dim)

            # Decoder part. NO FORCE TEACHING
            for t in range(max_len):
                # A check just to be compatible with no attention test
                if hasattr(decoder, 'attention'):
                    attention_weighted_encoding, _ = decoder.attention(encoder_out, h)
                    gate = decoder.sigmoid(decoder.f_beta(h))                                # gating scalar, (s, encoder_dim)
                    attention_weighted_encoding = gate * attention_weighted_encoding
                    h, c = decoder.decode_step(torch.cat([decoder.embedding(prev_word.squeeze(1)), attention_weighted_encoding], dim=1), (h, c))
                else:
                    gate = decoder.sigmoid(decoder.f_beta(h))  # gating scalar, (batch_size_t, encoder_dim)
                    h, c = decoder.decode_step(
                        torch.cat([decoder.embedding(prev_word.squeeze(1)), gate], dim=1),
                        (h, c))

                # Get the word has the highest score/probability and concat to a sequence
                scores = decoder.fc(h)                                                   # (s, vocab_size)
                prev_word = scores.max(1)[1].unsqueeze(1)
                seqs = torch.cat([seqs, prev_word], dim=1)

            # Hypothesis
            temp_preds = list()
            preds = seqs.tolist()                   # batch_size, max_length + 1
            for i in range(len(preds)):
                pred_no_pad = []
                j = 1
                while preds[i][j] != word_map['<end>']:
                    pred_no_pad.append(preds[i][j])
                    j += 1
                temp_preds.append(pred_no_pad)
            hypotheses.extend(temp_preds)

            # Reference
            for j in range(allcaps.shape[0]):
                img_caps = allcaps[j].tolist()
                img_captions = list(
                    map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<pad>'], word_map['<end>']}],
                        img_caps))  # remove <start> and pads
                references.append(img_captions)

            # test the correctness
            # for i in range(len(references)):
            #     refs_text = [reversed_word_map[num] for num in references[i][0]]
            #     hypotheses_text = [reversed_word_map[num] for num in hypotheses[i]]
            #     print(references[i][0])
            #     print(hypotheses[i])
            #     print(refs_text)
            #     print(hypotheses_text)
            #     ref_string = ' '.join(refs_text)
            #     hypotheses_string = ' '.join(hypotheses_text)
            #     print(hypotheses_string)
            #     print(ref_string)
            #     print()
            #     print()

        bleu4 = corpus_bleu(references, hypotheses)
        print(bleu4)
        return bleu4




def train_caption(train_loader, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer, epoch):
    """
    Performs one epoch's training.

    :param train_loader: DataLoader for training data
    :param encoder: encoder model
    :param decoder: decoder model
    :param criterion: loss layer
    :param encoder_optimizer: optimizer to update encoder's weights (if fine-tuning)
    :param decoder_optimizer: optimizer to update decoder's weights
    :param epoch: epoch number
    """

    decoder.train()  # train mode (dropout and batchnorm is used)
    encoder.train()

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss (per word decoded)
    top5accs = AverageMeter()  # top5 accuracy

    start = time.time()

    # Batches
    for i, (imgs, caps, caplens) in enumerate(train_loader):
        data_time.update(time.time() - start)

        # Move to GPU, if available
        imgs = imgs.to(device)
        caps = caps.to(device)
        caplens = caplens.to(device)

        # Forward prop.
        imgs = encoder(imgs)
        scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs, caps, caplens)

        # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
        targets = caps_sorted[:, 1:]

        # Remove timesteps that we didn't decode at, or are pads
        # pack_padded_sequence is an easy trick to do this

        # scores, _ = pack_padded_sequence(scores, decode_lengths, batch_first=True)
        scores = pack_padded_sequence(scores, decode_lengths, batch_first=True).data
        # targets, _ = pack_padded_sequence(targets, decode_lengths, batch_first=True)
        targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data

        # Calculate loss
        loss = criterion(scores, targets)

        # Add doubly stochastic attention regularization
        loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

        # Back prop.
        decoder_optimizer.zero_grad()
        if encoder_optimizer is not None:
            encoder_optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        if grad_clip is not None:
            clip_gradient(decoder_optimizer, grad_clip)
            if encoder_optimizer is not None:
                clip_gradient(encoder_optimizer, grad_clip)

        # Update weights
        decoder_optimizer.step()
        if encoder_optimizer is not None:
            encoder_optimizer.step()

        # Keep track of metrics
        top5 = accuracy_top_k(scores, targets, config.top_k)
        losses.update(loss.item(), sum(decode_lengths))
        top5accs.update(top5, sum(decode_lengths))
        batch_time.update(time.time() - start)

        start = time.time()

        # Print status
        if i % print_freq == 0:

            print_and_write('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})'.format(epoch, i, len(train_loader),
                                                                          batch_time=batch_time,
                                                                          data_time=data_time, loss=losses,
                                                                          top5=top5accs))


def validate_caption(val_loader, encoder, decoder, criterion, word_map):
    """
    Performs one epoch's validation.

    :param val_loader: DataLoader for validation data.
    :param encoder: encoder model
    :param decoder: decoder model
    :param criterion: loss layer
    :param word_map: a word map
    :return: BLEU-4 score
    """
    decoder.eval()  # eval mode (no dropout or batchnorm)
    if encoder is not None:
        encoder.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top5accs = AverageMeter()

    start = time.time()

    references = list()  # references (true captions) for calculating BLEU-4 score
    hypotheses = list()  # hypotheses (predictions)
    # reversed_word_map = {value: key for key, value in word_map.items()}

    # explicitly disable gradient calculation to avoid CUDA memory error
    # solves the issue #57
    with torch.no_grad():
        # Batches
        for i, (imgs, caps, caplens, allcaps) in enumerate(val_loader):

            # Move to device, if available
            imgs = imgs.to(device)
            caps = caps.to(device)
            caplens = caplens.to(device)

            # Forward prop.
            if encoder is not None:
                imgs = encoder(imgs)
            scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs, caps, caplens)

            # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
            targets = caps_sorted[:, 1:]

            # Remove timesteps that we didn't decode at, or are pads
            # pack_padded_sequence is an easy trick to do this
            scores_copy = scores.clone()
            scores = pack_padded_sequence(scores, decode_lengths, batch_first=True).data
            targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data

            # Calculate loss
            loss = criterion(scores, targets)

            # Add doubly stochastic attention regularization
            loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

            # Keep track of metrics
            losses.update(loss.item(), sum(decode_lengths))
            top5 = accuracy_top_k(scores, targets, config.top_k)
            top5accs.update(top5, sum(decode_lengths))
            batch_time.update(time.time() - start)

            start = time.time()

            if i % print_freq == 0:
                print_and_write('Validation: [{0}/{1}]\t'
                      'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})\t'.format(i, len(val_loader),
                                                                                batch_time=batch_time,
                                                                                loss=losses, top5=top5accs))

            # Store references (true captions), and hypothesis (prediction) for each image
            # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
            # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]

            # References

            sort_ind = sort_ind.cpu()
            allcaps = allcaps[sort_ind]  # because images were sorted in the decoder
            for j in range(allcaps.shape[0]):
                img_caps = allcaps[j].tolist()
                img_captions = list(
                    map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<pad>']}],
                        img_caps))  # remove <start> and pads
                references.append(img_captions)

            # Hypotheses
            _, preds = torch.max(scores_copy, dim=2)
            preds = preds.tolist()
            temp_preds = list()
            for j, p in enumerate(preds):
                temp_preds.append(preds[j][:decode_lengths[j]])  # remove pads
            preds = temp_preds
            hypotheses.extend(preds)

            assert len(references) == len(hypotheses)

        # test the correctness
        # for i in range(len(references)):
        #     refs_text = [reversed_word_map[num] for num in references[i][0]]
        #     hypotheses_text = [reversed_word_map[num] for num in hypotheses[i]]
        #     print(references[i][0])
        #     print(hypotheses[i])
        #     print(refs_text)
        #     print(hypotheses_text)
        #     ref_string = ' '.join(refs_text)
        #     hypotheses_string = ' '.join(hypotheses_text)
        #     print(hypotheses_string)
        #     print(ref_string)
        #     print()
        #     print()


        # Calculate BLEU-4 scores
        bleu4 = corpus_bleu(references, hypotheses)

        print_and_write(
            '\n * LOSS - {loss.avg:.3f}, TOP-5 ACCURACY - {top5.avg:.3f}, BLEU-4 - {bleu}\n'.format(
                loss=losses,
                top5=top5accs,
                bleu=bleu4))

    return bleu4


def save_checkpoint_caption(data_name, epoch, epochs_since_improvement, encoder, decoder, encoder_optimizer,
                            decoder_optimizer,
                            bleu4, is_best):
    """
    Saves model checkpoint.

    :param data_name: base name of processed dataset
    :param epoch: epoch number
    :param epochs_since_improvement: number of epochs since last improvement in BLEU-4 score
    :param encoder: encoder model (Be noted that this only save parameters only. USe load_state_dict to continue training)
    :param decoder: decoder model (Be noted that this only save parameters only. USe load_state_dict to continue training)
    :param encoder_optimizer: optimizer to update encoder's weights, if fine-tuning
    :param decoder_optimizer: optimizer to update decoder's weights
    :param bleu4: validation BLEU-4 score for this epoch
    :param is_best: is this checkpoint the best so far?
    """
    state = {'epoch': epoch,
             'epochs_since_improvement': epochs_since_improvement,
             'bleu-4': bleu4,
             'encoder': encoder.state_dict(),
             'decoder': decoder.state_dict(),
             'encoder_optimizer': encoder_optimizer,
             'decoder_optimizer': decoder_optimizer}
    filename = 'checkpoint_' + data_name + '.pth.tar'
    torch.save(state, 'model/' + filename)
    # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
    if is_best:
        torch.save(state, 'model/' + 'BEST_' + filename)


def print_and_write(message):
    print(message)
    with open(txt_name, 'a') as f:
        f.write(message + "\n")


def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.

    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def save_checkpoint(data_name, epoch, epochs_since_improvement, encoder, encoder_optimizer, best_acc, is_best):
    """
    Saves model checkpoint to Google Drive as default.

    :param data_name: base name of processed dataset
    :param epoch: epoch number
    :param epochs_since_improvement: number of epochs since last improvement in BLEU-4 score
    :param encoder: encoder model
    :param encoder_optimizer: optimizer to update encoder's weights, if fine-tuning
    :param is_best: is this checkpoint the best so far?
    """
    state = {'epoch': epoch,
             'epochs_since_improvement': epochs_since_improvement,
             'encoder': encoder.state_dict(),
             'best_acc': best_acc,
             'encoder_optimizer': encoder_optimizer, }
    filename = 'checkpoint_' + data_name + '.pth.tar'
    torch.save(state, 'model/' + filename)
    # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
    if is_best:
        torch.save(state, 'model/' + 'BEST_' + filename)


class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, shrink_factor):
    """
    Shrinks learning rate by a specified factor.

    :param optimizer: optimizer whose learning rate must be shrunk.
    :param shrink_factor: factor in interval (0, 1) to multiply learning rate with.
    """

    print_and_write("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    print_and_write("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))


def accuracy_top_k(scores, targets, k):
    """
    Computes top-k accuracy, from predicted and true labels.

    :param scores: scores from the model
    :param targets: true labels
    :param k: k in top-k accuracy
    :return: top-k accuracy
    """

    batch_size = targets.size(0)
    _, ind = scores.topk(k, 1, True, True)
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    correct_total = correct.view(-1).float().sum()  # 0D tensor
    return correct_total.item() * (100.0 / batch_size)


def get_number_parameter(model):
    params = sum([np.prod(p.size()) for p in model.parameters()])
    return params
