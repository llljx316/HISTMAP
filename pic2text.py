import torch
from PIL import Image
import open_clip

model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
model.eval()  # model in train mode by default, impacts some models with BatchNorm or stochastic depth active
tokenizer = open_clip.get_tokenizer('ViT-B-32')

image = preprocess(Image.open("segres/3431780/12.png")).unsqueeze(0)
descriptions = [
    # Geographical Features
    "A vast and uncharted ocean with sea monsters drawn along the edges",
    "A mountain range said to be the home of ancient gods",
    "A dense forest known as the 'Forbidden Woods' with mythological creatures",
    "A great desert with ancient caravan routes",
    "A massive inland sea surrounded by ancient cities",
    "A mighty river, the lifeblood of the surrounding kingdom",
    "A series of small islands labeled as unexplored territories",
    "A volcanic island rumored to be cursed",
    "A deep chasm believed to lead to the underworld",
    "A large lake where ancient rituals were performed",

    # Ancient Civilizations and Kingdoms
    "The grand capital of an ancient empire with towering walls",
    "A legendary lost city deep in the jungle",
    "A walled fortress at the edge of the known world",
    "A nomadic settlement along the trade route",
    "The ruins of a once-great civilization, now overgrown with vines",
    "An ancient temple dedicated to a forgotten god",
    "A royal palace, home to a long lineage of kings",
    "A mysterious kingdom beyond the mountains, marked by symbols",
    "A sacred burial ground of ancient rulers",
    "A coastal trading port, bustling with merchants from distant lands",

    # Structures and Landmarks
    "A grand stone lighthouse guiding ancient mariners",
    "A colossal statue marking the border of the kingdom",
    "A stone bridge built by a legendary architect spanning a great river",
    "An aqueduct bringing water from distant mountains to the city",
    "A watchtower guarding the mountain pass",
    "An ancient library said to hold the wisdom of the ages",
    "A pyramid or ziggurat, the site of religious ceremonies",
    "A maze-like palace, home to an ancient king",
    "A monolithic altar where offerings were made to the gods",
    "A sacred obelisk marking the victory of an ancient battle",

    # Trade Routes and Roads
    "The Silk Road, a well-trodden path connecting distant empires",
    "A caravan trail passing through treacherous deserts",
    "The King’s Highway, an ancient road lined with milestones",
    "A sea route marked by the locations of ancient trading ports",
    "A pilgrimage path leading to a sacred temple",
    "An ancient stone road, now overgrown and abandoned",
    "The spice route across the eastern seas",
    "A coastal route frequented by pirates",
    "A forgotten road leading to a hidden valley",
    "A river trade route used by merchants from faraway lands",

    # Mythological or Mystical Elements
    "A mysterious island, home to a legendary sea serpent",
    "A forbidden mountain said to be the home of dragons",
    "A cursed forest where travelers disappear without a trace",
    "A portal to the underworld hidden in a deep cave",
    "The realm of the gods, high atop a distant peak",
    "A sacred grove, watched over by ancient spirits",
    "A vast whirlpool in the ocean, believed to be the mouth of a sea god",
    "A floating city, home to sorcerers and alchemists",
    "An enchanted castle surrounded by an impenetrable mist",
    "The ruins of a temple where oracles once predicted the future",

    # Maritime Elements and Navigation
    "A sea route dotted with ancient lighthouses guiding sailors home",
    "An ancient harbor, filled with ships bearing the sigils of long-gone empires",
    "A treacherous strait marked with illustrations of shipwrecks",
    "A mysterious archipelago known only to a few skilled navigators",
    "A whirling maelstrom feared by all sailors",
    "A dangerous reef, home to sirens and other sea creatures",
    "A sea marked as 'Here Be Dragons' near the edge of the map",
    "The path of a famous explorer’s voyage into the unknown",
    "A port marked as a hub for pirate activity",
    "The trade winds, crucial for ancient sea voyages",

    # Ancient Cities and Towns
    "A bustling port city, the gateway to the eastern lands",
    "A hilltop fortress overlooking the surrounding plains",
    "A city-state renowned for its philosophers and scholars",
    "An ancient village nestled in the shadow of a volcano",
    "A market town at the crossroads of several trade routes",
    "The capital of a long-forgotten kingdom, now in ruins",
    "A riverside settlement known for its ancient pottery",
    "A city famed for its towering pyramids and golden tombs",
    "A mountain village, isolated from the world for centuries",
    "A fortified town known for repelling countless invasions",

    # Borders and Territories
    "The Great Wall separating the civilized world from barbarian lands",
    "A contested border region marked with ancient battlefields",
    "The frontier, beyond which no explorer has returned",
    "A kingdom’s boundary marked by a series of ancient watchtowers",
    "A natural border formed by a river, dividing two rival kingdoms",
    "A no-man’s-land between warring empires",
    "A forbidden zone, said to be cursed by the gods",
    "The lands of the nomads, stretching into the horizon",
    "A wilderness inhabited by legendary creatures",
    "A disputed coastline where ships from different nations battle for control",

    # Sacred or Religious Sites
    "A mountaintop shrine where ancient priests once performed rituals",
    "A temple built into the cliffs, dedicated to a sea god",
    "A sacred grove where druids once held ceremonies",
    "A cave covered in ancient paintings, believed to be holy",
    "A stone circle used as a calendar by ancient astronomers",
    "The ruins of a cathedral, destroyed by a great calamity",
    "A desert temple, said to hold the secrets of eternal life",
    "A pilgrimage site visited by travelers from distant lands",
    "An ancient monastery perched on a hilltop",
    "The tomb of a forgotten king, guarded by statues of warriors",

    # Symbols and Cartographic Elements
    "A compass rose at the corner of the map, marking the cardinal directions",
    "A legend explaining the various symbols used on the map",
    "An illustration of the sun and moon, representing time and cycles",
    "A dragon or serpent depicted on the edges of the known world",
    "A map scale represented by a ruler or series of notches",
    "Intricate borders decorated with mythological creatures",
    "Drawings of ships sailing the seas, with flags indicating their origin",
    "A depiction of a wind god blowing across the map, representing the trade winds",
    "Arrows showing the path of ancient explorers or conquerors",
    "Illustrations of animals native to the different regions on the map"
]

text = tokenizer(descriptions)

with torch.no_grad(), torch.cuda.amp.autocast():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

print("Label probs:", text_probs)  # prints: [[1., 0., 0.]]
print(f"most probably description: {descriptions[torch.argmax(text_probs).item()]}")