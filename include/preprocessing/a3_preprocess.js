const splitEvery = (n, list) => {
    const result = [];
    let idx = 0;
    while (idx < list.length) {
        result.push(list.slice(idx, (idx += n)));
    }
    return result;
};

const fs = require('fs');
const filename = 'news';
const file = fs.readFileSync(filename, 'utf8');
const lines = file.split('\n');
const categoriesWithTitle = splitEvery(8, lines).map(infos => [infos[6], infos[0]].map(s => s.trim()).join(' '));
fs.writeFileSync('corpora', categoriesWithTitle.join('\n'), 'utf8');
