import re

def read_messages(filename):
    with open(filename, 'r') as f:
        content = f.read()
    return [msg.strip() for msg in content.split('@@') if msg.strip()]

def read_bics(filename):
    with open(filename, 'r') as f:
        return [line.strip() for line in f if line.strip()]

def extract_message_type(message):
    match = re.search(r'\{2:([A-Z])(\d{3}).*?\}', message)
    if match:
        return match.group(2)
    else:
        raise ValueError("Message type not found in header.")

def update_message(message, new_bic, sequence_num):
    # Insert terminal code 'A' in the 9th position (new_bic is 11 characters)
    if len(new_bic) == 11:
        new_bic = new_bic[:8] + 'A' + new_bic[8:]

    # Update Receiver BIC in block {1:}
    message = re.sub(r'(\{1:.{4})(.{12})(.*?\})', r'\1' + new_bic + r'\3', message)

    # Extract message type again (in case it changed during substitution)
    msg_type = extract_message_type(message)
    seme_value = f"MT{msg_type}SWP{sequence_num:08d}"

    # Update or insert :20C::SEME// tag
    if ":20C::SEME//" in message:
        message = re.sub(r':20C::SEME//.*', f':20C::SEME//{seme_value}', message)
    else:
        message += f"\n:20C::SEME//{seme_value}"

    return message

def process_messages(input_file, bic_file, output_file):
    messages = read_messages(input_file)
    bics = read_bics(bic_file)

    output_messages = []
    sequence_num = 1

    for bic in bics:
        for msg in messages:
            updated_msg = update_message(msg, bic, sequence_num)
            output_messages.append(updated_msg)
            sequence_num += 1

    with open(output_file, 'w') as f:
        for msg in output_messages:
            f.write(f"@@{msg}\n")

# Example usage:
# process_messages('input_mt_messages.txt', 'bic_list.txt', 'output_mt_messages.txt')
